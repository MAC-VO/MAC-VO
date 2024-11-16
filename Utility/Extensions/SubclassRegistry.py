from typing import TypeVar
from ..PrettyPrint import Logger


T = TypeVar('T', bound='SubclassRegistry')


class SubclassRegistry(object):
    """
    Provides ability to instantiate subclass (and grandchilds) dynamically by name at runtime.
    
    `cls.instantiate(name: str, *args, **kwargs) -> T`
    """
    __HIERARCHY: dict[str, type] = dict()
    
    @classmethod
    def name(cls) -> str:
        """
        Assign a short name for the dataset class. By default will be the class name.
        Overwrite this function if you want to create a more readable name used in `type` field in config.
        """
        return cls.__name__
    
    @classmethod
    def instantiate(cls: type[T], type: str, *args, **kwargs) -> T:
        return cls.get_class(type)(*args, **kwargs)
    
    @classmethod
    def get_class(cls: type[T], type: str) -> type[T]:
        clsname = type  # bad to have variable name as keyword, but the most fitting name
        if clsname in cls.__HIERARCHY: return cls.__HIERARCHY[clsname]
        Logger.write("fatal", f"Get '{clsname}' from class {cls.__name__}, expect to be one of {list(cls.__HIERARCHY.keys())}")
        raise KeyError(f"Get '{clsname}' from class {cls.__name__}, expect to be one of {list(cls.__HIERARCHY.keys())}")
    
    def __init_subclass__(cls, **kwargs) -> None:
        cls.__HIERARCHY = {"": cls}
        checkbase = list(filter(lambda x: issubclass(x, SubclassRegistry), cls.__bases__))
        assert len(checkbase) == 1, "Does not support diamond inheritance in SubclassRegistry"
        
        for pcls in cls.mro()[1:]:
            if not issubclass(pcls, SubclassRegistry): continue
            if cls.name() in pcls.__HIERARCHY:
                Logger.write("fatal", f"SubclassRegistry Error: There more than one descendent of class '{pcls.__name__}' with name of {cls.name()}. "
                             "This introduces ambiguity to dynamic reflection and is therefore disallowed.")
                raise NameError(f"SubclassRegistry Error: There more than one descendent of class '{pcls.__name__}' with name of {cls.name()}. "
                             "This introduces ambiguity to dynamic reflection and is therefore disallowed.")
            else:
                pcls.__HIERARCHY[cls.name()] = cls
