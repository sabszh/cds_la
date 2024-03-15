import pandas

def test():
    print('Hello world!')
    return pandas.DataFrame({'a': [1, 2], 'b': [3, 4]})

if __name__ == '__main__':
    test()