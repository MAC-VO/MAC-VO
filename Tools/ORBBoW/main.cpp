#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openssl/sha.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "include/ORBVocabulary.h"

namespace {

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    return fields;
}

std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read vocabulary");
    SHA256_CTX context;
    SHA256_Init(&context);
    std::vector<char> buffer(1 << 20);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) SHA256_Update(&context, buffer.data(), count);
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &context);
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned char byte : digest) result << std::setw(2) << static_cast<int>(byte);
    return result.str();
}

std::vector<cv::Mat> rows(const cv::Mat& descriptors) {
    std::vector<cv::Mat> result;
    result.reserve(descriptors.rows);
    for (int index = 0; index < descriptors.rows; ++index) {
        result.push_back(descriptors.row(index));
    }
    return result;
}

struct Candidate {
    long frame_id;
    double score;
};

struct StoredORBFrame {
    DBoW2::BowVector bow;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct SparseMatchResult {
    int raw_knn_count = 0;
    int ratio_count = 0;
    std::vector<cv::DMatch> matches;
};

SparseMatchResult match_features(
    const StoredORBFrame& source,
    const cv::Mat& target_descriptors,
    double ratio,
    int max_matches
) {
    SparseMatchResult result;
    if (source.descriptors.empty() || target_descriptors.empty()) return result;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> forward_knn;
    matcher.knnMatch(source.descriptors, target_descriptors, forward_knn, 2);
    result.raw_knn_count = static_cast<int>(forward_knn.size());

    std::vector<cv::DMatch> ratio_matches;
    ratio_matches.reserve(forward_knn.size());
    for (const auto& pair : forward_knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < ratio * pair[1].distance) {
            ratio_matches.push_back(pair[0]);
        }
    }
    result.ratio_count = static_cast<int>(ratio_matches.size());

    std::vector<cv::DMatch> reverse;
    matcher.match(target_descriptors, source.descriptors, reverse);
    for (const auto& match : ratio_matches) {
        if (match.trainIdx < 0 || match.trainIdx >= static_cast<int>(reverse.size())) continue;
        const auto& backward = reverse[match.trainIdx];
        if (backward.trainIdx == match.queryIdx) result.matches.push_back(match);
    }
    std::sort(result.matches.begin(), result.matches.end(), [](const cv::DMatch& left, const cv::DMatch& right) {
        if (left.distance != right.distance) return left.distance < right.distance;
        if (left.queryIdx != right.queryIdx) return left.queryIdx < right.queryIdx;
        return left.trainIdx < right.trainIdx;
    });
    if (static_cast<int>(result.matches.size()) > max_matches) {
        result.matches.resize(max_matches);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    std::string vocabulary_path;
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--vocabulary") {
            vocabulary_path = argv[index + 1];
            ++index;
        }
    }
    if (vocabulary_path.empty()) {
        std::cerr << "--vocabulary is required\n";
        return 2;
    }

    ORB_SLAM3::ORBVocabulary vocabulary;
    if (!vocabulary.loadFromTextFile(vocabulary_path)) {
        std::cerr << "failed to load vocabulary\n";
        return 3;
    }
    auto orb = cv::ORB::create(1200);
    std::unordered_map<long, StoredORBFrame> database;
    std::cout << "READY\t2\t" << sha256_file(vocabulary_path)
              << "\t" << CV_VERSION << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "STOP") {
            std::cout << "BYE" << std::endl;
            return 0;
        }
        if (line == "PING") {
            std::cout << "PONG" << std::endl;
            continue;
        }
        try {
            const auto fields = split_tabs(line);
            if (fields.size() != 8 || fields[0] != "QUERY") {
                throw std::runtime_error("expected QUERY with seven arguments");
            }
            const long request_id = std::stol(fields[1]);
            const long target_id = std::stol(fields[2]);
            const std::string image_path = fields[3];
            const long min_gap = std::stol(fields[4]);
            const int top_k = std::stoi(fields[5]);
            const double ratio = std::stod(fields[6]);
            const int max_matches = std::stoi(fields[7]);
            if (request_id < 0 || target_id < 0 || min_gap < 0 || top_k < 1 ||
                !std::isfinite(ratio) || ratio <= 0.0 || ratio >= 1.0 || max_matches < 1) {
                throw std::runtime_error("numeric arguments are invalid");
            }
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (image.empty()) throw std::runtime_error("unable to read image");
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
            DBoW2::BowVector current;
            if (!descriptors.empty()) vocabulary.transform(rows(descriptors), current);

            std::vector<Candidate> candidates;
            for (const auto& item : database) {
                if (target_id - item.first < min_gap) continue;
                candidates.push_back({item.first, vocabulary.score(current, item.second.bow)});
            }
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
                if (left.score != right.score) return left.score > right.score;
                return left.frame_id < right.frame_id;
            });
            if (static_cast<int>(candidates.size()) > top_k) candidates.resize(top_k);
            std::cout << "RESULT\t" << request_id << "\t" << target_id
                      << "\t" << candidates.size();
            std::cout << std::setprecision(17);
            for (const auto& candidate : candidates) {
                const auto& source = database.at(candidate.frame_id);
                const auto matched = match_features(
                    source, descriptors, ratio, max_matches
                );
                std::cout << "\t" << candidate.frame_id << ":" << candidate.score
                          << ":" << matched.raw_knn_count
                          << ":" << matched.ratio_count
                          << ":" << matched.matches.size() << ":";
                for (std::size_t match_index = 0; match_index < matched.matches.size(); ++match_index) {
                    if (match_index) std::cout << ";";
                    const auto& match = matched.matches[match_index];
                    const auto& source_keypoint = source.keypoints.at(match.queryIdx);
                    const auto& target_keypoint = keypoints.at(match.trainIdx);
                    std::cout << source_keypoint.pt.x << "," << source_keypoint.pt.y
                              << "," << target_keypoint.pt.x << "," << target_keypoint.pt.y
                              << "," << match.distance;
                }
            }
            std::cout << std::endl;
            database[target_id] = StoredORBFrame{
                current, keypoints, descriptors.clone()
            };
        } catch (const std::exception& error) {
            std::cout << "ERROR\t" << error.what() << std::endl;
        }
    }
    return 0;
}
