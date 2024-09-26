from typing import TypeVar
from ..PrettyPrint import Logger


T = TypeVar('T', bound='SubclassRegistry')


class SubclassRegistry(object):
    """
    Provides ability to instantiate subclass (and grandchilds) dynamically by name at runtime.
    
    `cls.instantiate(name: str, *args, **kwargs) -> T`
    """
    __HIERARCHY: dict[str, type] = dict()
    __ABSPATH: str = ""
    
    @classmethod
    def name(cls) -> str:
        """
        Assign a short name for the dataset class. By default will be the class name.
        Overwrite this function if you want to create a more readable name used in `name` field in config.
        """
        return cls.__name__
    
    @classmethod
    def instantiate(cls: type[T], type: str, *args, **kwargs) -> T:
        clsname = type  # bad to have variable name as keyword, but the most fitting name
        name2 = "/" + clsname
        if clsname not in cls.__HIERARCHY and name2 not in cls.__HIERARCHY:
            Logger.write("fatal", f"Get '{clsname}', expect to be one of {list(cls.__HIERARCHY.keys())}")
            raise KeyError()
        else:
            if clsname in cls.__HIERARCHY: return cls.__HIERARCHY[clsname](*args, **kwargs)
            else: return cls.__HIERARCHY[name2](*args, **kwargs)
    
    @classmethod
    def get_class(cls: type[T], type: str) -> type[T]:
        clsname = type  # bad to have variable name as keyword, but the most fitting name
        name2 = "/" + clsname
        if clsname in cls.__HIERARCHY: return cls.__HIERARCHY[clsname]
        elif name2 in cls.__HIERARCHY: return cls.__HIERARCHY[name2]
        Logger.write("fatal", f"Get '{clsname}', expect to be one of {list(cls.__HIERARCHY.keys())}")
        raise KeyError()
    
    def __init_subclass__(cls, **kwargs) -> None:
        cls.__HIERARCHY = {"": cls}
        checkbase = list(filter(lambda x: issubclass(x, SubclassRegistry), cls.__bases__))
        assert len(checkbase) == 1, "Does not support diamond inheritance in RegisterClassTree"
        
        direct_pcls = checkbase[0]
        cls.__ABSPATH = direct_pcls.__ABSPATH + "/" + cls.name()
        
        for pcls in cls.mro()[1:]:
            if not issubclass(pcls, SubclassRegistry): continue
            pcls.__HIERARCHY[cls.__ABSPATH.replace(pcls.__ABSPATH, "", 1)] = cls
