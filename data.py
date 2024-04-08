import pandas as pd

# 紫金矿业 江西铜业 铜陵有色 西部矿业 云南铜业
# 601899 600362 000630 601168 000878
# CU9999
pair = {}
pair['CU9999.XSGE'] = ['601899', '600362', '000630', '601168', '000878']

# 潞安环能 山西焦煤 永泰能源 淮北矿业 华阳股份
# 601699 000983 600157 600985 600348
# JM9999
pair['JM9999.XDCE'] = ['601699', '000983', '600157', '600985', '600348']

# 宝钢股份 包钢股份 华菱钢铁 首钢股份 鞍钢股份
# 600019 600010 000932 000959 000898
# HC9999
pair['HC9999.XSGE'] = ['600019', '600010', '000932', '000959', '000898']

# 中国石油 中国石化 荣盛石化 恒力石化 东方盛虹
# 601857 600028 002493 600346 000301
# SC9999
pair['SC9999.XINE'] = ['601857', '600028', '002493', '600346', '000301']

# 三友化工 吉林化纤 新乡化纤 恒天海龙 南京化纤
# 600409 000420 000949 000677 600889
# TA9999
pair['TA9999.XZCE'] = ['600409', '000420', '000949', '000677', '600889']

# 宝丰能源 鲁西化工 华谊集团 诚志股份 江苏索普
# 600989 000830 600623 000990 600746
# L9999
pair['L9999.XDCE'] = ['600989', '000830', '600623', '000990', '600746']

# 赛轮轮胎 玲珑轮胎 三角轮胎 通用股份 贵州轮胎
# 601058 601966 601163 601500 000589
# RU9999
pair['RU9999.XSGE'] = ['601058', '601966', '601163', '601500', '000589']

# 中粮糖业 冠农股份 南宁糖业 华资实业
# 600737 600251 000911 600191
# SR9999
pair['SR9999.XZCE'] = ['600737', '600251', '000911', '600191']

# 大北农 天康生物 唐人神 禾丰股份 傲农生物
# 002385 002100 002567 603609 603363
# M9999
pair['M9999.XDCE'] = ['002385', '002100', '002567', '603609', '603363']

# 百隆东方 鲁泰 A 华孚时尚 孚日股份 华茂股份
# 601339 000726 002042 002083 000850
# CF9999
pair['CF9999.XZCE'] = ['601339', '000726', '002042', '002083', '000850']

fut_list = ['CU9999.XSGE', 'JM9999.XDCE', 'HC9999.XSGE', 'SC9999.XINE', 'TA9999.XZCE', 'L9999.XDCE', 'RU9999.XSGE', 'SR9999.XZCE', 'M9999.XDCE', 'CF9999.XZCE']



metal = pd.read_csv("stocks/metal.csv")
metal['Stkcd'] = metal['Stkcd'].astype(str).str.zfill(6)

coal = pd.read_csv("stocks/coal.csv")
coal['Stkcd'] = coal['Stkcd'].astype(str).str.zfill(6)

nonferrous = pd.read_csv("stocks/nonferrous.csv")
nonferrous['Stkcd'] = nonferrous['Stkcd'].astype(str).str.zfill(6)

all_stock = pd.concat([metal, coal, nonferrous], axis=0)
all_stock.rename(columns={'Trddt': 'date', 'Opnprc': 'open','Loprc':'low', 'Hiprc':'high', 'Clsprc':'close'}, inplace=True)
all_stock['date'] = pd.to_datetime(all_stock['date'])
all_stock['open_diff'] = all_stock['open'].diff()
all_stock['high_diff'] = all_stock['high'].diff()
all_stock['low_diff'] = all_stock['low'].diff()
all_stock['close_diff'] = all_stock['close'].diff()
all_stock['label'] = all_stock['ChangeRatio']>0



def fut_read(x):
    fut = pd.read_csv(f"futures/{x}.csv")
    fut['date'] = pd.to_datetime(fut['date'])
    fut['ChangeRatio'] = fut['close'].diff() / fut['close'].shift(1) 
    fut['ChangeRatio'].iloc[0] = 0
    fut['open_diff'] = fut['open'].diff()
    fut['high_diff'] = fut['high'].diff()
    fut['low_diff'] = fut['low'].diff()
    fut['close_diff'] = fut['close'].diff()
    fut = fut[fut.date>='2019-05-01']
    return fut

def stock_read(x):
    dfs = []
    stock_df = []
    stock_codes = pair[x]
    # print(len(stock_codes))
    for code in stock_codes:
        df = all_stock[all_stock["Stkcd"] == code]
        df = df[df.date>='2019-05-01']
        stock_df.append(df)
    return stock_df
# 注意！有些期货对应的股票没数据。

if __name__ == '__main__':
    pass