# encoding=utf-8
import jieba
import thulac
import re


def cut_and_filter(string):
    """
    给句子分词，并只保留汉字，返回定长的词的列表
    """
    # 全模式，试图最精确地分词，成词后还将细分
    #wordgen = jieba.cut(str, cut_all=True)
    # 精确模式，能成词即分开，不考虑细分
    wordgen = jieba.cut(string, cut_all=False)
    # 精确模式对于商品名称分词比较适用，因为商品品牌/型号/参数不需要分割，而最代表商品本质的词语一般不需要再细分

    words = []
    for word in wordgen:
        # 若词只包含中文
        if re.match(r'^[\u4e00-\u9fa5]+$', word) is not None:
            words.append(word)

    return words


if __name__ == "__main__":
    '''
    thu1 = thulac.thulac(seg_only=True)
    words4 = thu1.cut("胜鑫款珠宝 Y1601001767 18K金镶钻紫罗兰翡翠转运珠吊坠女款玉坠", text=True)
    words5 = thu1.cut("我想买一个DVD播放器。", text=True)
    '''
    w2 = cut_and_filter("ansevi(安视威) IC卡/M1卡/门禁卡/考勤卡/异形卡 蓝色IC方牌")
    w3 = cut_and_filter("胜鑫款珠宝 Y1601001767 18K金镶钻紫罗兰翡翠转运珠吊坠女款玉坠")
    w4 = cut_and_filter('腾讯QQ黑钻贵族九个月 地下城勇士黑钻9个月 包月卡可查时间可续费★自动充值')
    w5 = cut_and_filter('SENMA/森马2016夏季新品情侣运动休闲鞋圆头系带鞋男女鞋舒适板鞋 男-蓝色 42')
    w6 = cut_and_filter('YIDUO|壹朵 荷花系列LED吸顶灯 客厅卧室餐厅灯饰 现代简约个性铁艺无极遥控调光灯具 YXD-016 花语 36W 正白光')
    print(w2)
    print(w3)
    print(w4)
    print(w5)
    print(w6)
