import json
def check(file1,file2,save_path):
    with open('dataset/word_dictionary.json', "r") as fh:
        word2id = json.load(fh)
        word2id['-LRB-']=word2id['(']
        word2id['-RRB-']=word2id[')']
        word2id['\'\'']=word2id['"']
        word2id['``']=word2id['"']
        word2id['-LSB-']=word2id['[']
        word2id['-RSB-']=word2id[']']
    sens=[]
    with open(file1,'r') as f1:
        while True:
            x=f1.readline()
            if not x:
                break
            sens.append(x)

    correct_num=0
    wrong_num=0
    avg_max_f1=0
    safe_sen=[]
    with open(file2,'r') as f2:
        while True:
            x=f2.readline()
            if not x:
                break
            j=int(x.split(' ')[-1])

            # 条件1：长度相同
            x_=x.split(')')[0:-1]
            x_=list(filter(lambda x:x!='',x_))
            print(j, len(x_), len(sens[j].split(' ')))

            # 条件2：词汇全部包含
            words = list(map(lambda x:x.split(' ')[-1],x_))
            contain_all=True
            for w in words:
                if w not in word2id.keys():
                    print(w)
                    contain_all=False
                    break

            # 条件3：最大f1==1
            x_2 = x.split('(')[1:]
            x_2 = list(map(lambda x: float(x.split(' ')[0]), x_2))
            mx_f1 = max(x_2)
            avg_max_f1 += mx_f1
            if len(x_)==len(sens[j].split(' ')) and mx_f1==1 and contain_all:
                correct_num+=1
                safe_sen.append(x)
            else:
                wrong_num+=1

    all_num=correct_num+wrong_num
    avg_max_f1/=all_num

    print('total:',all_num)
    print('correct_num:',correct_num)
    print('wrong:',wrong_num)
    print('avg_max_f1:',avg_max_f1)

    with open(save_path,'w') as f3:
        for s in safe_sen:
            f3.write(s)

data_type='dev'
check('trees/'+data_type+'_word_sen.txt','trees/'+data_type+'_sen2.txt','trees/'+data_type+'_sen3.txt')

