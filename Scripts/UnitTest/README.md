# UnitTest

This directory stores all the unit test cases for this repository.

* Use `@pytest.marker.local` to decorate a test function if you don't want to run it 
(either too slow or impossible) on CI/CD like GitHub Action.

* Use `@pytest.marker.trt` to decorate a test function if it requires TensorRT to run properly.

To run all test cases, go to root directory of this repository, and run `pytest`.
