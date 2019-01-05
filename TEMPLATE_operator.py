# -*- coding: utf-8 -*-
# 一个理想的operator模板


def opDemo(*args, **kwargs):
    '''
    args: var0-class arguments, len(args)= var0Num
    kwargs: other parameters, key = value.
    '''
    A = args[0]
    B = args[1]

    return A+B

def add(A, B, method='max'):
    
    return A+B