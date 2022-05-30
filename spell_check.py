from hanspell import spell_checker
import pandas as pd
import re
from tqdm import tqdm

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\\s]", "", sent)
    return sent_clean

def spell_check(text):
    text = clean_text(text)
    if len(text):
        return spell_checker.check(text).checked
    else:
        return text

def main(data_path, out_path, sep='|', encoding='euc-kr'):
    print(data_path)
    data = pd.read_csv(data_path, sep=sep, encoding=encoding)
    
    for c, col in enumerate(['text_obj','text_mthd','text_deal']):
        c += 4
        print(col)
        
        # 1. 
        data[col] = data[col].fillna('')
        for r in tqdm(range(len(data)), total=len(data)):
            try:
                data.iloc[r, c] = spell_check(data.iloc[r, c])
            except:
                continue
        
# #         # 2. 
# #         data[col]=data[col].fillna('').apply(spell_check)

#         # 3.
#         rng = 100000
#         for r in tqdm(range(0, len(data), rng), total=len(data)/rng):
#             clean_na = data[col].fillna('')
#             selected = clean_na[r:r+rng].to_list()
#             hsp_checked = spell_checker.check(selected)
#             hsp_checked = list(map(lambda x: x.checked, hsp_checked))
#             data.iloc[r:r+rng, c] = hsp_checked

    # save
    try:
        data.to_csv(out_path+'.txt', header=True, index=False, sep=sep, encoding=encoding)
        data.to_csv(out_path+'.csv', header=True, index=False, encoding=encoding)
    except:
        data.to_csv(out_path+'.txt', header=True, index=False, sep=sep, encoding='utf-8')
        data.to_csv(out_path+'.csv', header=True, index=False, encoding='utf-8')
    else:
        import pdb
        pdb.set_trace()
        data.to_csv(out_path+'.txt', header=True, index=False, sep=sep, encoding='utf-8')
        data.to_csv(out_path+'.csv', header=True, index=False, encoding='utf-8')
        
if __name__=='__main__':
    #main('/home/jupyter/2. 모델개발용자료.txt', '/home/jupyter/data_prep/2. 모델개발용자료_hsp')
    main('/home/jupyter/1. 실습용자료.txt', '/home/jupyter/data_prep/1. 실습용자료_중복제거_hsp')