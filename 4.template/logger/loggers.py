instances = {}  # 众多实例对象

"""
自定义日志包
"""


def classLogger(cls):
    """
    类日志装饰器，用于输出类init过程
    :param cls: 当前类
    :return: 日志信息
    """

    def get_instance(*args, **kwargs):
        """
        获取类的实例对象
        :param args:
        :param kwargs:
        :return:
        """
        cls_name = cls.__name__
        print(f"[Class]: {cls_name} is init...")
        if not cls_name in instances:
            instance = cls(*args, **kwargs)
            instances[cls_name] = instance
        print(f"[Class]: {cls_name} has inited.")
        return instances[cls_name]

    return get_instance


def funcLogger(func):
    """
    函数日志装饰器，用于查看训练日志
    :param func: call方法
    :return: 返回包装对象
    """
    func_name = func.__name__.strip("__")

    def wrapper(*args, **kwargs):
        """
        包装函数
        :param args:
        :param kwargs:
        :return: 日志信息
        """
        print(f"[{func_name}]: It's is training...")
        func(*args, **kwargs)
        print(f"[{func_name}]: Training has finished.")

    return wrapper
