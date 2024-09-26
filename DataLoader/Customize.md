# Dataset Customization

## Create a new Dataset (Sequence)

All the DIY dataset should inherit the `GenericSequence`. Create one new file in under `./DataLodaer`. 

The `GenericSequence` class is an abstract base class designed to manage datasets in a structured way, enabling easy extension and integration with various types of sequence data. Below is a detailed guide on how to implement and use this class for your datasets.

``` python
from .SequenceBase import GenericSequence


class MySequence(GenericSequence[SomeDataType]):
    @classmethod
    def name(cls) -> str: return "MySeqName"    # Optional
    
    def __getitem__(self, local_index: int) -> SomeDataType:
        # logical index is the index exposed to user and may be affected by multiple 
        # cropping and masking process
        #
        # index is the "real" index / the index of dataset before all cropping and
        # masking operations.
        index = self.get_index(local_index)

        # Implementation for loading a frame, **Required**
        return SomeDataType(...)
```

Requirements

* `__getitem__` **Method (Required):** This is an abstract method that must be implemented to load individual frames from the dataset. It returns a DataFrame for the specified index. You can customize your own DataFrame or use the default provided by the library. Custom data frame format must be subclass of `DataFrame` 

* `name` **Method (Optional):** This method can be overridden to return a custom name for your dataset type. If left blank, it will default to the class name.

* Include your dataset in `DataLodaer/__init__.py`. When you are creating a new yaml file for this dataset, fill the `name` with the class name or the short name you defined in the `name`.

### Creating a New DataFrame

The `DataFrame` class, located in `Interface.py`, is designed as a base class for data handling, particularly for automating data collation. All new data types that need to be managed by the `GenericSequence` use this class.

For example, if you need to handle camera data, which typically includes only RGB images, you can create a new calss called `MyDataFrame`. This class inherits from `DataFrame` and requires minimal additional code if it only handles data types such as `torch.Tensor`, `pp.LieTensor`, and `np.ndarray` that are directly supported by the default `collate` method in `DataFrame`. Here is an example:


```python
from .Interface import DataFrame

class MyDataFrame(DataFrame):
    def __init__(self, image: torch.Tensor):
        self.image = image
    # No need to override the collate method if only handling supported data types
```

**Overriding the Collate Method**

   If you want to create your own collation logic or if you want to handle several attributes collectively, you may choose to override the `collate` method directly.

   Example
   ```python
   from .Interface import DataFrame

   class MyDataFrame2(DataFrame):
       def __init__(self, image: torch.Tensor, some_random_other_data: str):
           self.other_data = other_data
           self.image = image
       
       @classmethod
       def collate(cls, batch: list[Self]) -> Self:
           # Collate images and other data together
           return cls(
               torch.cat([x.image for x in batch], dim=0),
               "|".join([x.other_data for x in batch])
           )
   ```

This structure allows for flexibility in handling various types of data and ensures that the system can efficiently process batches of data, regardless of their specific attributes or the complexity of the data types involved.



## Usage

Here’s how you can instantiate and use your dataset class:

``` python
dataset = GenericSequence.instantiate("MyDatasetName", root="/path/to/dataset", K=torch.eye(3))
# Equivalent to the following if name(cls) -> str is not implemented by `MyDatasetName`
dataset = MyDatasetName(root="/path/to/dataset", K=torch.eye(3))
dataloader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)

for batch in dataloader:
    print(batch)
```
