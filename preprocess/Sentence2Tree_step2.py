from tqdm import tqdm
import numpy as np

# 获取f1 score
def get_f1(pred,gt):
    tp=len(list(set(pred).intersection(set(gt))))
    precision=tp/len(pred)
    recall=tp/len(gt)
    if precision+recall==0:
        return 0
    else:
        return 2*(precision*recall)/(precision+recall)

# 获取em score
def get_em(pred,gt):
    if pred[0]==gt[0] and pred[-1]==gt[-1]:
        return 1
    else:
        return 0

# 去除树多余的节点
def remove_d(sen):
    ex={}
    del_num=0
    sen2=sen.copy()
    hit_dict={}
    index=0
    for i,s in enumerate(sen):
        if s!='(0 ':
            hit_dict[i]=index
            index+=1
    for i in range(len(sen)):
        if sen[i]=='(0 ':
            ln = 0
            rn = 0
            hit_list = []
            for j in range(i, len(sen)):
                if sen[j] == '(0 ':
                    ln += 1
                else:
                    hit_list.append(str(hit_dict[j]))
                    rn += sen[j].count(')')
                    if rn >= ln:
                        if '_'.join(hit_list) not in ex:
                            ex['_'.join(hit_list)] = 1
                            break
                        else:
                            sen2[i]='delete(0 '
                            sen2[j]=sen2[j].replace(')','',1)
                            del_num += 1
                            break
    for i in range(del_num):
        sen2.remove('delete(0 ')
    return sen2

# 给树的每个节点给分(label:f1)
def gself(sen0,gt):
    sen=[]
    temp = ""
    for s in sen0:
        if len(temp)>0 and s=='(':
            sen.append(temp)
            temp=""
        temp+=s
        if temp=='(0 ':
            sen.append(temp)
            temp=""
    sen.append(temp)
    sen=remove_d(sen)
    hit_dict={}
    index=0
    for i,s in enumerate(sen):
        if s!='(0 ':
            hit_dict[i]=index
            index+=1
    for i in range(len(sen)):
        if sen[i]=='(0 ':
            ln=0
            rn=0
            hit_list=[]
            for j in range(i,len(sen)):
                if sen[j]=='(0 ':
                    ln+=1
                else:
                    hit_list.append(hit_dict[j])
                    rn+=sen[j].count(')')
                    if rn>=ln:
                        f1=get_em(hit_list,gt)
                        sen[i]='('+str(f1)+' '
                        break
    sen=''.join(sen)
    # print(sen)
    return sen

# 文本预处理
def preprocess(sen,type=0):
    if type==0:# 转化前预处理
        sen = sen.replace('--OOV--', 'UNKNOWN')
    else:# 转化后处理
        sen = sen.replace('UNKNOWN', '--OOV--')
        sen = sen.strip()
    return sen

# 将句子转化为树格式(未去重复节点，未标注)
def tree_transpose(ori_path, save_path, save_path_fin, label_path):
    num=0
    gts = np.load(label_path)
    wrong_list=[]
    with open(save_path_fin,'w') as fn:
        with open(save_path,'w') as f:
            with open(ori_path,'r') as fh:
                while True:
                    error=False
                    num+=1
                    print('transpose:',num)
                    x=fh.readline()
                    if not x:
                        break
                    sen = x

                    # pre0
                    sen=preprocess(sen,type=0)
                    try:
                        sen=nlp.parse(sen)
                        sen=str(sen).split('\n')
                        for i,s in enumerate(sen):
                            sen[i]=s.lstrip()+" "
                        sen=''.join(sen)

                        # pre1
                        sen=preprocess(sen,type=1)
                        f.write(sen)
                        f.write('\n')
                    except:
                        error=True
                        wrong_list.append(num)
                        print('wrong in',num)

                    if not error:
                        gt = gts[num-1, 0:2]
                        gt = np.arange(gt[0], gt[1] + 1)
                        sen = sen.split(')')
                        for i, s in enumerate(sen):
                            s_temp = s.strip().split(' ')
                            sen[i] = ""
                            for j in range(len(s_temp) - 1):
                                if j == 0 and i != 0:
                                    sen[i] += " "
                                sen[i] += '(0 '
                            sen[i] += s_temp[-1]
                        # label zero clean
                        sen = ')'.join(sen)
                        sen = gself(sen, gt)
                        fn.write(sen)
                        fn.write('\n')

    return wrong_list

# 将句子转化为树格式(未去重复节点，未标注)
def tree_transpose0(ori_path, save_path_fin, label_path):
    gts = np.load(label_path)
    wrong_list=[]
    with open(save_path_fin,'w') as fn:
            with open(ori_path,'r') as fh:
                while True:
                    x=fh.readline()
                    if not x:
                        break
                    sen = x
                    num = int(sen.split(' ')[-1])
                    sen = ' '.join(sen.split(' ')[0:-1]).strip()
                    print('transpose:',num)
                    gt = gts[num, 0:2]
                    gt = np.arange(gt[0], gt[1] + 1)
                    sen = sen.split(')')
                    for i, s in enumerate(sen):
                        s_temp = s.strip().split(' ')
                        sen[i] = ""
                        for j in range(len(s_temp) - 1):
                            if j == 0 and i != 0:
                                sen[i] += " "
                            sen[i] += '(0 '
                        sen[i] += s_temp[-1]
                    # label zero clean
                    sen = ')'.join(sen)
                    sen = gself(sen, gt)
                    fn.write(sen)
                    fn.write(' ' + str(num))
                    fn.write('\n')

    return wrong_list

wrong_list=tree_transpose0('../dataset/dev_sen0.txt','../dataset/dev_sen2.txt','../dataset/dev_y_true2.npy')