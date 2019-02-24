# -*- coding: utf-8 -*-
"""
mxue, Jan. 2019
This module try to build a factor expression class
""" 

import os
import yaml
import random
import numpy as np
from copy import deepcopy
from functools import partial
import multiprocessing

# anytree, see https://anytree.readthedocs.io/en/2.4.3/
from anytree import Node
from anytree import RenderTree
from anytree import AnyNode

import utils

class Alphabet(object):

    def __init__(self, opConfigFile=None, **kwargs):
        # config file for operators
        self.opConfigFile = opConfigFile
        # 'rand', 'enum', 'mimic'
        self.mode = kwargs.get('mode', 'mimic')

        # strategy template which define hierachical relation of ops.
        self.template = kwargs.get('template', None)

        self.allOps = None
        self.opGraph = None
        self.opVarsLib = None

    def getAllOps_(self, opConfigFile=None, **kwargs):
        """get all involved operators
        (i)  pass in directly.
        (ii) from yaml file.
        """
        prnt = kwargs.get('prnt', False)
 
        if opConfigFile is not None:
            allOps = utils.read_yml(opConfigFile) 
        elif self.opConfigFile is not None:
            allOps = utils.read_yml(self.opConfigFile, prnt=prnt)
        else:
            print('ERROR: operator config. file not found...')
            raise IOError

        return allOps

    def opLib2opGraph_(self, *args, **kwargs):
        """build relation btw. operator by two methods
        i) by its `info` segment in .yaml config file.
        ii) by a given template which specify the nested relation btw. operators.
        """

        n_proc = kwargs.get('n_proc',4)

        if self.allOps is not None:
            keys = self.allOps.keys() # all opName iterator
        else:
            self.getAllOps_()
            keys = self.allOps.keys() # all opName iterator
        
        # Two type of opGraph    
        if self.mode == 'mimic':
            # build graph from TEMPLATE
            if self.template is not None:
                tree = utils.build_tree_from_template(self.template)
            elif kwargs.get('template') is not None:
                tree = utils.build_tree_from_template(kwargs.get('template'))
            else:
                print('No template file to mimic.')
                raise IOError

            operator_graph = tree

        elif (self.mode == 'rand')|(self.mode == 'enum'):
            # apply_async for all operators
            pool = multiprocessing.Pool(n_proc)
            res = []
            for op in keys:
                # for each op
                res.append(pool.apply_async(utils.find_available_nested_op, args=(op, self.allOps),kwds={'mode':self.mode}))
            pool.close()
            pool.join()

            # collect results into operator_graph
            operator_graph = {}
            for i in res:
                operator_graph.update(i.get())
        else:
            print("Unknown mode, check 'cls().mode'...")
            raise ValueError

        return operator_graph

    def opLib2opVarsLib_(self, *args, **kwargs):
        """build all possible combinations of op vars"""
        # multiprocess， apply_async:
        #      10 ops around 14000 para options, 300ms at jupterNotebook.
        # apply_async for all operators
        
        n_proc = kwargs.get('n_proc', 4)

        pool = multiprocessing.Pool(n_proc)
        res = []
        for op in self.allOps.keys():
            # for each op
            res.append(pool.apply_async(utils.build_paraList_for_ONE_op_dict, args=(op, self.allOps)))
        pool.close()
        pool.join()

        # collect results into operator_graph
        opVarsLib = {}
        for i in res:
            opVarsLib.update(i.get())

        return opVarsLib

    def buildOpLib(self, *args, **kwargs):
        # prnt
        self.allOps = self.getAllOps_(prnt=kwargs.get('prnt', False))
        # n_proc
        self.opGraph = self.opLib2opGraph_(n_proc=kwargs.get('n_proc',4),**kwargs)
        self.opVarsLib = self.opLib2opVarsLib_(n_proc=kwargs.get('n_proc',4))
        return None

    def enumerate_(self,):
        """枚举生成opTree
        i) 给定模板的话，对模板的每个op的参数进行枚举
        ii) 没有给定模板，就按照opGraph和opVars暴力枚举所有可能的算符匹配，所有算符参数可能性，两者可独立做
        """
        if self.mode=='mimic':
            # 有模板的枚举参数
            tree_list = utils.enumerate_opVars_for_tree(self.opGraph, self.opVarsLib)
        else:
            # 无模板的枚举算符组合，枚举参数
            print('This is too brutal !')
            ptree_list = None
        return tree_list

    def random_(self, mode=None, complexity=None, N=1):
        """枚举生成opTree
        i) 给定模板的话，对模板的每个op的参数进行随机选择安装
        ii) 没有给定模板，就按照opGraph和opVars随机枚举所有可能的算符匹配，所有算符参数可能性
        NOTE: return is the root of Operators Tree
        """
        if self.mode=='mimic':
            # 需要指定N，也即返回几个随机生成表达式，模板就是opGraph
            tree_list = utils.generate_N_tree_by_template(N, self.allOps, self.opGraph, self.opVarsLib)

        else:
            # 需要指定complexity和N
            tree_list = utils.generate_N_tree_given_complexity(N, complexity, self.allOps, self.opGraph, self.opVarsLib)
    
        return tree_list


# ---------- 1. 生成策略树（根据opGraph 和mode） -----------------------------------
    # @classmethod
    def tree2formula(self, tree_might_list):
        """ 
        tree_might_list: root of tree, or a tree list.
        return: formula expression, or formula expressions list.
        """
        # pdb.set_trace()
        if isinstance(tree_might_list, list)|isinstance(tree_might_list, np.ndarray):
            ret = [utils.tree2formula(node) for node in tree_might_list]
        else:
            ret =  utils.tree2formula(tree_might_list) 
        return ret

    def replace_op(self, root, opVars):
        """
        root: an opTree;  
        rVars: specified opVars for ops to replace.
        NOTE: operators alternatives are regarded as equivalent with old ops.
        EQUIVLANT: operators with the same number of var0 variables.
        """
        # 0. 
        pass

    # see utils
    def replace_opVar(self, root, rVars):
        # 换算符，枚举参数
        pass

if __name__ == '__main__':
    import pdb
    import pandas as pd
    # a = Alphabet('./opTestNaive.yml', prnt=True, mode='mimic')
    # a.buildOpLib(template='./anynodeTree.yml')

    a = Alphabet('./op.yml', prnt=True, mode='mimic')
    a.buildOpLib(template='./tree.yml')
    # pdb.set_trace()
    # tree = a.random_(N=4)
    print(RenderTree(a.opGraph))

    # pdb.set_trace()
    
    tree_list = a.enumerate_()
    formula_list = a.tree2formula(tree_list)
    df = pd.DataFrame(formula_list, columns=['strg'])
    df.to_csv('./exprs_.csv')
    
    # pdb.set_trace()
