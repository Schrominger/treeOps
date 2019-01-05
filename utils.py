# -*- coding: utf-8 -*-
# 2018. Nov., mxue

import os
import yaml
import random
import multiprocessing
import numpy as np
from copy import deepcopy
from functools import partial
# anytree, see https://anytree.readthedocs.io/en/2.4.3/
from anytree import Node
from anytree import RenderTree
from anytree import AnyNode


def check_dimension_for_tree(root):
    node = deepcopy(root)
    try:
        opDim = allOps[node.name]['info'][0][3]
        var0List =
        var0Dim = [allVars[v]['dim'] for v in var0List]
        if someMatchCondition(opDim, var0Dim):
            op = op.parent
            opDim = allOps[node.name].output.dim
            check_dimension_for_tree
        else:
            return False
    except:
        return True


def select_op_given_complexity(operator_config_yml, N, mode='random'):
    # calculate the operator relation Graph.
    operator_graph = generate_graph_for_operators_lib(operator_config_yml)
    # read all operators infomation.
    allOps = read_yml(operator_config_yml)

    # FORMULA is is a structure: [{[{...}]}]
    FORMULA = []
    # initilize the operators pool, always start with operator complexity=1
    initilized_pools = [['fillna']] #

    COMPLEXITY = 0

    while (COMPLEXITY < N):
        if COMPLEXITY==0:
            current_op_pools = initilized_pools

        next_op_pools = []
        for op_pool in current_op_pools:
           
            current_op = random.choice(op_pool)
   
            op_para_list = build_paraList_for_ONE_op(current_op, operator_config_dict)

            op_para_dct = random.choice(op_para_list) # 只要var1,var2

            var0Num = len(allOps[current_op]['input']['var0'])
            FORMULA = [{}]*var0Num + [[op_para_dct]]

            COMPLEXITY = COMPLEXITY + var0Num
            if COMPLEXITY >= N:

                for i in range(var0Num):
                    value = 0 #### var0的变量
                    FORMULA[i] = {'identity':value}
                pass
            else:
                # fill in nested operators at location of var0

                # build op_pool ,select one
                pass

            # setup parameters for current op
            tmp_op = install_para_for_op(current_op, para)
            # build next pool
            next_op_pools = next_op_pools.append(operator_graph[current_op])

            options_pools = build_op_pool(current_op)

    formula_exp_list = map(utils.dict_list_to_expr, formula_list)
    return formula_exp_list


def select_op_given_complexity2(operator_config_yml, N, treeMode='random', paraMode='enumerate'):

    # 0. build a formula tree of ops
    if treeMode == 'random':
        formula_tree = random_build_tree(N, allOps)
    elif treeMode == 'breadth':
        formula_tree = breadth_first_build_tree(N, allOps)
    elif treeMode == 'depth':
        formula_tree = depth_first_build_tree(N, allOps)

    # 1. add parameters for the ops tree.
    formula_list = formula_tree_to_formula(formula_tree, allOps, paraMode=paraMode)

    return formula_list

def random_build_tree(N, allOps):
    pass

def breadth_first_build_tree(N, opGraph):
    # root = Node('fillna')
    # current_root = (root,)
    for rt in current_root:
        children_nodes = [Node(n) for n in opGraph[rt.name]]
        rt.children = children_nodes
        COMPLEXITY += len(children_nodes)
        if COMPLEXITY < N:
            current_root = rt.children
        else:
            pass
    return root

def depth_first_build_tree(N, allOps):
    pass


def generate_N_expr_given_complexity(N, complexity, op_config):

    allOps, opGraph, opVarsLib = build_opLibrary(op_config)
    print('Finish oplibrary!')
    root = AnyNode(name='fillna', A='Close')
    def recursionBuildTree(j):
        return tree_to_formula(recursion_to_build_tree(deepcopy(root), complexity, allOps, opGraph, opVarsLib))
    expr_list = map(recursionBuildTree, range(N))

    print('--- Generate %s formula !' %N)
    return expr_list

def build_opLibrary(op_config, prnt = False):
    allOps = read_yml(op_config, prnt=prnt)
    opGraph = opLib_to_opGraph(allOps)
    opVarsLib = opLib_to_opVarsLib(allOps)
    return allOps, opGraph, opVarsLib


#  nice !!!!
def tree_to_formula(node):
    ''' TREE TO FORMULA
    recursion is great !
    code is short, thinking time is long ...
    '''
    EXPR = ''
    if node.is_leaf:
        EXPR = node_to_expr(node)
    else:
        EXPR = node_to_expr_var0_placeholder(node)
    # nice map and format cooperation!!!
    EXPR = EXPR.format(*map(tree_to_formula, node.children))
    return EXPR

 # 更漂亮的版本
def tree_to_formula2(node):
    EXPR = node_to_expr(node) if node.is_leaf else node_to_expr_var0_placeholder(node)
    EXPR = EXPR.format(*map(tree_to_formula2, node.children))
    return EXPR

def node_to_expr(node):
    # do not destroy original node
    node_copy = deepcopy(node)
    # the list you want to filter to *args rather than **kwargs inputs.
    ref_set = ['A','B','C','D','E']
    # pop name
    opName = node_copy.__dict__.pop('name')
    # pop var0
    var0_list = [node_copy.__dict__.pop(j) for j in ref_set if j in node_copy.__dict__.keys()]

    expr = opName + '(' + ','.join(var0_list) + ',' + str(node_copy)[8:-1] + ')'
    if len(str(node_copy)[8:-1])==0:
        expr = opName + '(' + ','.join(var0_list) + ')'

    return expr

def node_to_expr_var0_placeholder(node):
    # do not destroy original node
    node_copy = deepcopy(node)
    # the list you want to filter to *args rather than **kwargs inputs.
    ref_set = ['A','B','C','D','E'] # maybe you need to put allTradeDate in this list.
    # pop name
    opName = node_copy.__dict__.pop('name')
    # pop var0
    var0_list = [node_copy.__dict__.pop(j) for j in ref_set if j in node_copy.__dict__.keys()]
    # placeholder
    var0_placeholder = ['{}']*len(var0_list)

    expr = opName + '(' + ','.join(var0_placeholder) + ',' + str(node_copy)[8:-1] + ')'
    if len(str(node_copy)[8:-1])==0:
        expr = opName + '(' + ','.join(var0_placeholder)  + ')'

    return expr

def recursion_to_build_tree(root, maxCnt, allOps, opGraph, opVarsLib):
    if len(root.children)==0 and maxCnt>len(root.siblings):
        # find suitable chidren
        children_iterator = get_children_iterator(root.name, allOps, opGraph, opVarsLib)
        root, _ = addNode(root, children_iterator)

    if not check_overbear(root, maxCnt):
        numChild = len(root.children)
        # 遍历子节点
        for node in root.children:
            node = recursion_to_build_tree(node, (maxCnt-numChild), allOps, opGraph, opVarsLib)
    return root


def get_children_iterator(opName, allOps, opGraph, opVarsLib):
    # choose children ops
    chosePara_list = []
    for i in range(len(allOps[opName]['input']['var0'])):
        chosedOp = random.choice(opGraph[opName]) # get one op
        chosePara = random.choice(opVarsLib[chosedOp]) # get para for this op
        chosePara['name'] = chosedOp # merge op's name to para dict
        chosePara_list.append(chosePara)

    return chosePara_list

def addNode(rootNode, children_iterator):
    # children_iterator = [{},{},...], dictionary iterator
    for child in children_iterator:
        AnyNode(**child, parent=rootNode)
    return rootNode, rootNode.children

def check_overbear(node, planNum):
    '''
    planNum： 计划生育数
    node： 被检测的节点
    '''
    return len(node.descendants)>planNum


def opLib_to_opGraph(allOps, n_processor=3):
    '''
    就是generate_graph_for_operators_lib()
    '''
    keys = allOps.keys() # all opName iterator

    # apply_async for all operators
    pool = multiprocessing.Pool(n_processor)
    res = []
    for op in keys:
        # for each op
        res.append(pool.apply_async(find_available_nested_op, args=(op, allOps)))
    pool.close()
    pool.join()

    # collect results into operator_graph
    operator_graph = {}
    for i in res:
        operator_graph.update(i.get())

    return operator_graph

# (finihsed)
def generate_graph_for_operators_lib(operator_config_yml, dump=False, n_processor = 3):

    allOps = read_yml(operator_config_yml)
    keys = allOps.keys() # all opName iterator

    # apply_async for all operators
    pool = multiprocessing.Pool(n_processor)
    res = []
    for op in keys:
        # for each op
        res.append(pool.apply_async(find_available_nested_op, args=(op, allOps)))
    pool.close()
    pool.join()

    # collect results into operator_graph
    operator_graph = {}
    for i in res:
        operator_graph.update(i.get())

    return operator_graph


def find_available_nested_op(op, allOps):
    '''find available ops for nest, return dict, value is a list.'''
    current_opInfo = allOps[op]['info']
    ret_list = []
    for candidate in allOps.keys():
        candidate_info = allOps[candidate]['info']
        if some_match_rule_for_opInfo(current_opInfo[0], candidate_info[1]):
            ret_list.append(candidate)

    retDict = {op:ret_list}
    # ------------------------------
    # Future issues: 是否需要考虑滤重
    # -----------------------------
    return retDict

# ======================= HOW TO MATCH TWO OPERATORS ? =========================
def some_match_rule_for_opInfo(a, b):
    '''define some rule for op match
    input: a, b for comparision
    return: True or False
    '''
    return int(str(a)[0]) >= int(str(b)[0])

def some_match_rule_for_opInfo2(a, b):
    '''define some rule for op match
    检查前后算符的量纲是否有包含关系
    return: True or False
    '''
    return int(str(a)[0]) >= int(str(b)[0])
# =============================================================================
# add more control segment, sucn as for dimension matching or dimension mapping
# inside operators
# =============================================================================
# build paraList for a given operator from the config-file dict.
def opLib_to_opVarsLib(allOps, n_processor=3):
    # multiprocess， apply_async:
    #      10 ops around 14000 para options, 300ms at jupterNotebook.
    # apply_async for all operators
    pool = multiprocessing.Pool(n_processor)
    res = []
    for op in allOps.keys():
        # for each op
        res.append(pool.apply_async(build_paraList_for_ONE_op_dict, args=(op, allOps)))
    pool.close()
    pool.join()

    # collect results into operator_graph
    opVarsLib = {}
    for i in res:
        opVarsLib.update(i.get())

    return opVarsLib

# this is a wrapper
def build_paraList_for_ONE_op_dict(op, allOps):
    return {op:build_paraList_for_ONE_op(op, allOps)}

def build_paraList_for_ONE_op(op, allOps):
    if allOps[op].get('input'):
        dct = allOps[op]['input']
    else:
        # for test,
        dct = allOps

    dctValueisArray = {}
    # collect var0,var1,var2, all parameters list
    for key in dct.keys():
        if dct[key] is not None:
            dctValueisArray.update(dct[key])

    op_para_option_list = split_options_map_dict(dctValueisArray)
    return  op_para_option_list

def dict_list_to_expr2(dct_ls):

    EXPR = ''
    for cnt, dct in enumerate(dct_ls):
        if isinstance(dct, dict):
            # print(dct)
            if cnt != len(dct_ls)-1:
                EXPR = EXPR + peel_dict_to_insert(dct)+','
            else:
                EXPR = EXPR + peel_dict_to_insert(dct)
            key = list(dct.keys())[0]
            if isinstance(dct[key],list):
                EXPR = EXPR.format(dict_list_to_expr2(dct[key]))
        elif isinstance(dct, list):
            # list in list, parameter in this layer
            if cnt != len(dct_ls)-1:
                EXPR = EXPR + para_dict_to_str(dct)+','
            else:
                EXPR = EXPR + para_dict_to_str(dct)
    return EXPR

def para_dict_to_str(ls_of_dct):

    if len(ls_of_dct)==0:
        return ''

    paraList = []
    for dct in ls_of_dct:
        if len(dct)>0:
            for key, value in dct.items():
                paraList.append("{0}={1}".format(key,value))

    para_list_string = ','.join(paraList)
    return para_list_string


def generate_from_given_template(confName):
    '''
    Input: config file name
    Ouput: genrate formula by replacement on given template.
    '''
    # read template
    y = read_yml(confName)
    y_options, y_template = y['OPTIONS'], y['TEMPLATE']

    # split y_options to single-dict list.
    y_option_list = split_options_map_dict(y_options)
    # print('THE ORIGINAL TEMPLATE:\n', yaml.dump(y_template))
    # generate based on the template, powerful map !
    y_FORMULA_list = map(partial(replace_template_by_dict, y_template), y_option_list)
    return list(y_FORMULA_list)

def split_options_map_dict(dct_valisArray):

    tmpList = [[]]*len(list(dct_valisArray.keys()))
    # build list of list.
    cnt = 0
    for key in dct_valisArray.keys():
        tmpList[cnt] = [{key:v} for v in dct_valisArray[key] ]
        cnt += 1

    # filter the empty list
    tmpList = [tmpls for tmpls in tmpList if len(tmpls)>0]

    # lists combinations
    dct_ls = lists_combination(tmpList)
    return dct_ls

def lists_combination(lists):
    '''
    输入多个列表组成的列表, 输出其中每个列表所有元素可能的所有排列组合
    python3以上 reduce需要导入
    '''
    import sys

    if sys.version_info > (3, 1):
        try:
            import reduce
        except:
            from functools import reduce

    def myfunc(ls1, ls2):
        return [{**i,**j} for i in ls1 for j in ls2]
    return reduce(myfunc, lists)

# python3
def merge_two_dict(dct1,dct2):
    return {**dct1, **dct2}


def replace_template_by_dict(template, repMap):

    from copy import deepcopy

    ret = deepcopy(template) # deepcopy
    for cnt,dct in enumerate(ret):
        key = list(dct.keys())[0]
         # 0. replace if needed
        if key in repMap.keys():
            ret[cnt] = replace_key_for_dct(dct, repMap[key], key=key)
            key = repMap[key]
        # 1. recursive process
        if isinstance(ret[cnt][key],list):
            ret[cnt][key] = replace_template_by_dict(ret[cnt][key], repMap)
    return ret

def replace_key_for_dct(dct, repl, **kwargs):

    # the key need to replace
    key = kwargs.get('key', list(dct.keys())[0])
    # new key = repl
    dct[repl] = dct.pop(key)
    return dct


def peel_dict_to_insert(dct):
    '''
    input: dct ={key:value}
    if value is a list: return key({})
    else: return key(value)
    '''
    # only deal with the first key !
    key = list(dct.keys())[0]
    value = dct[key] # value is a list contains dicts or a value !

    expr = str(key)+'('+ '{}' +')' # key({})
    if not isinstance(value, list):
        expr = expr.format(value)
    return expr

def dict_list_to_expr(dct_ls):
    EXPR = ''
    for cnt, dct in enumerate(dct_ls):
        if cnt != len(dct_ls)-1:
            EXPR = EXPR + peel_dict_to_insert(dct)+','
        else:
            EXPR = EXPR + peel_dict_to_insert(dct)
        key = list(dct.keys())[0]
        if isinstance(dct[key],list):
            EXPR = EXPR.format(dict_list_to_expr(dct[key]))

    return EXPR

# reverse ？
def expr_to_dict_list_to(EXPR):
    pass


# 读入yaml配置文件
def read_yml(fileName, filePath='./configs', block=None, prnt=False):
    '''
    read .yml file from given path.
    '''
    # default: get the whole block
    if fileName.endswith('.yml'):
        filePathName = os.path.join(filePath, fileName)
    else:
        filePathName = os.path.join(filePath, fileName+'.yml')
    # 此处可以增加try，except，尝试打开文件
    f = open(filePathName, encoding='UTF-8')
    y = yaml.load_all(f)
    y = list(y)

    if len(y)==1:
        # only one block
        y_prnt = y[0]
    else:
        if block is not None:
            # choose the specific block
            y_prnt = y[block] if y else None # return None if empty
        else:
            y_prnt = y

    if prnt:
        print(yaml.dump(y_prnt))
    return y_prnt


if __name__ == '__main__':
    import numpy as np
    import yaml

    ls = [{'add':[{'add':[{'substract':[{'identity':'a'},{'identity':'b'}]},
              {'divide':[{'identity':'a'},{'identity':'b'}]}]},
              {'multiply':[{'identity':'a'},{'identity':'b'}]} ]}]

    def add(x,y):
        return x+y
    def substract(x,y):
        return x-y
    def multiply(x,y):
        return x*y
    def divide(x,y):
        return x/y if y!=0. else np.nan
    def identity(x):
        return x

    print ('In the format of yaml:\n')
    print (yaml.dump(ls))

    a=2.; b =3.

    EXPR = dict_list_to_expr(ls)
    print(EXPR)

    eval(EXPR)
