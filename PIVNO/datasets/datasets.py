import copy


datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

# datasets[dataset_spec['name']](**dataset_args) 是实例化类的方式。它通过名称从 datasets 字典中查找类，然后使用字典 dataset_args 中的参数来实例化类。
def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
