import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import pandas as pd
def get_doc_term_list():
    docs = []

    for i in range(1, 9):
        for j in range(1, 1001):
            f2 = open('E:/paper/split_result/train/' + str(i) + '/' + str(j) + '.txt', 'r')
            text1 = f2.read().strip(',').decode('utf-8').split(',')

            docs.append(text1)
            f2.close()
    return docs


def get_term_dict(doc_terms_list):
    term_set_dict = {}

    for doc_terms in doc_terms_list:

        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())
    print term_set_list[11827]
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict,term_set_list

def stats_term_class_df(doc_terms_list, term_dict):
    term_class_df_mat = np.zeros((len(term_dict), 8), np.float32)

    for k in range(0,8000):
        class_index = k/1000
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat


def feature_selection_mi(term_class_df_mat,term_set):
    class_df_list =[1000,1000,1000,1000,1000,1000,1000,1000]
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)

    term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size)))

    term_score_max_list = [max(x) for x in term_score_mat]

    term_score_array = np.array(term_score_max_list)
    print term_score_array

    return term_score_array


if __name__ == '__main__':

    doc_term_list = get_doc_term_list()
    term_dict,term_list = get_term_dict(doc_term_list)
    term_class_df = stats_term_class_df(doc_term_list,term_dict)
    term_score_array = feature_selection_mi(term_class_df,term_list)

    data = pd.DataFrame(term_score_array,index=term_list)
    data.to_csv('E:\\paper\\mi\\orgin.csv', header=False)