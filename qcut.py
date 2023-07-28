import numpy as np
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.multicomp import MultiComparison
import warnings
warnings.filterwarnings("ignore")

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv(r'C:\Users\ahmed.demirel\PycharmProjects\BizimToptanSecond\escut()_function\flo_data_20k.csv')
df = df_.copy()


df["TotalBuy"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

today_date = dt.datetime(year=2021, month=12, day=31)
today_date = pd.to_datetime(today_date)


rfm = df[["master_id", "TotalBuy", "TotalPrice", "last_order_date"]]
rfm["recency"] = [(today_date - col).days for col in rfm["last_order_date"]]
rfm = rfm.drop("last_order_date", axis=1)
rfm = rfm[["master_id", "recency", "TotalBuy", "TotalPrice"]]
rfm.columns = ["master_id", 'recency', 'frequency', 'monetary']
rfm = rfm[rfm["monetary"] > 0]

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması


def isqcut_ok(series,q=5, ):
    if q in [3, 4, 5, 6]:
        temp_series = pd.qcut(series.rank(method="first"), q, labels=[str(col) + ". Grup" for col in range(1, q + 1)])
        temp_df = pd.DataFrame({"score": list(series), "group": list(temp_series)})
        print(temp_df.groupby("group").agg({"count", "mean", "median"}))
        print("***********************************")
        print(color.BOLD + color.BLUE + "Shapiro ile Normallik Testi Sonuçları:" + color.END)
        groups = []
        pvalues = []
        for group in list(temp_df.group.unique()):
            groups.append(temp_df.loc[temp_df["group"] == group, "score"])
            pvalue = shapiro(temp_df.loc[temp_df["group"] == group, "score"].values)[1]
            pvalues.append(pvalue)
            print(group, 'p-value: %.3f' % pvalue)
        print("***********************************")
        if any(i < 0.05 for i in pvalues):
            print(color.BOLD + color.CYAN + "Normalik Kontrolü Yapıldı, Gruplar Normal Dağılmıyor.. Non-Parametric Test Yapılacak.."+ color.END)
            if q == 3:
                test_value, pvalue1 = kruskal(groups[0], groups[1], groups[2])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue1))
            elif q == 4:
                test_value, pvalue1 = kruskal(groups[0], groups[1], groups[2], groups[3])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue1))
            elif q == 5:
                test_value, pvalue1 = kruskal(groups[0], groups[1], groups[2], groups[3], groups[4])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue1))
            elif q == 6:
                test_value, pvalue1 = kruskal(groups[0], groups[1], groups[2], groups[3], groups[4], groups[5])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue1))
            else:
                pass

            if pvalue1 < 0.05:
                print(color.BOLD + color.CYAN + "Bu grupların ortalaması arasında istatistiksel olarak anlamlı bir farklılık vardır." + color.END)
                comparison = MultiComparison(temp_df["score"], temp_df["group"])
                tukey = comparison.tukeyhsd(0.05)
                print(tukey)
                if any(i == False for i in tukey.reject):
                    print(color.BOLD + color.CYAN + "Gruplara bakıldığında aralarında anlamlı bir farklılık vardır." + color.END)
                    print(color.BOLD + color.RED + "Fakat gruplar kendi arasında değerlendirildiğinde bazı grupların birbirlerinden farkı olmadığı gözlemlenmiştir." + color.END)
                    print(color.BOLD + color.RED + "Bundan dolayı grup sayısında değişikliğe gidebilir veya veri setinizi gruplamaya bunu göze alarak devam edebilirsiniz." + color.END)
                else:
                    print(
                        color.BOLD + color.GREEN + "Bu diziyi {} gruba ayırmak istatistiksel olarak anlamlıdır. Kullanabilirsiniz. ".format(q) + color.END)

            else:
                print(color.BOLD + color.RED + "Bu grupların ortalaması arasında istatistiksel olarak anlamlı bir farklılık yoktur." + color.END)

        else:
            print(color.BOLD + color.CYAN + "Normalik Kontrolü Yapıldı, Gruplar Normal Dağılıyor.. Varyans Homojenliği Kontrol Edilecek.." + color.END)
            if q == 3:
                test_value, pvalue2 = levene(groups[0], groups[1], groups[2])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue2))
            elif q == 4:
                test_value, pvalue2 = levene(groups[0], groups[1], groups[2], groups[3])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue2))
            elif q == 5:
                test_value, pvalue2 = levene(groups[0], groups[1], groups[2], groups[3], groups[4])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue2))
            elif q == 6:
                test_value, pvalue2 = levene(groups[0], groups[1], groups[2], groups[3], groups[4], groups[5])
                print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue2))
            else:
                pass

            if pvalue2 > 0.05:
                print(color.BOLD + color.CYAN + "Varyans Homojenliği Kontrolü Yapıldı, Grupların Varyansı Homojen.. Parametric Test Yapılacak.." + color.END)
                if q == 3:
                    test_value, pvalue3 = f_oneway(groups[0], groups[1], groups[2])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                elif q == 4:
                    test_value, pvalue3 = f_oneway(groups[0], groups[1], groups[2], groups[3])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                elif q == 5:
                    test_value, pvalue3 = f_oneway(groups[0], groups[1], groups[2], groups[3], groups[4])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                elif q == 6:
                    test_value, pvalue3 = f_oneway(groups[0], groups[1], groups[2], groups[3], groups[4], groups[5])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                else:
                    pass

                if pvalue3 < 0.05:
                    print(
                        color.BOLD + color.GREEN + "Bu diziyi {} gruba ayırmak istatistiksel olarak anlamlıdır. Kullanabilirsiniz. ".format(
                            q) + color.END)
                else:
                    print(color.BOLD + color.RED + "Bu grupların ortalaması arasında istatistiksel olarak anlamlı bir farklılık yoktur." + color.END)


            else:
                print(color.BOLD + color.CYAN + "Varyans Homojenliği Kontrolü Yapıldı, Grupların Varyansı Homojen Değil.. Non-Parametric Test Yapılacak.." + color.END)
                if q == 3:
                    test_value, pvalue3 = kruskal(groups[0], groups[1], groups[2])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                elif q == 4:
                    test_value, pvalue3 = kruskal(groups[0], groups[1], groups[2], groups[3])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                elif q == 5:
                    test_value, pvalue3 = kruskal(groups[0], groups[1], groups[2], groups[3], groups[4])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                elif q == 6:
                    test_value, pvalue3 = kruskal(groups[0], groups[1], groups[2], groups[3], groups[4], groups[5])
                    print('Test Stat = %.3f, p-value = %.3f' % (test_value, pvalue3))
                else:
                    pass

                if pvalue3 < 0.05:
                    print(color.BOLD + color.CYAN + "Bu grupların ortalaması arasında istatistiksel olarak anlamlı bir farklılık vardır." + color.END)
                    comparison = MultiComparison(temp_df["score"], temp_df["group"])
                    tukey = comparison.tukeyhsd(0.05)
                    print(tukey)
                    if any(i == False for i in tukey.reject):
                        print(
                            color.BOLD + color.CYAN + "Gruplara bakıldığında aralarında anlamlı bir farklılık vardır." + color.END)
                        print(
                            color.BOLD + color.RED + "Fakat gruplar kendi arasında değerlendirildiğinde bazı grupların birbirlerinden farkı olmadığı gözlemlenmiştir." + color.END)
                        print(
                            color.BOLD + color.RED + "Bundan dolayı grup sayısında değişikliğe gidebilir veya veri setinizi gruplamaya bunu göze alarak devam edebilirsiniz." + color.END)
                    else:
                        print(
                            color.BOLD + color.GREEN + "Bu diziyi {} gruba ayırmak istatistiksel olarak anlamlıdır. Kullanabilirsiniz. ".format(
                                q) + color.END)

                else:
                    print(color.BOLD + color.RED + "Bu grupların ortalaması arasında istatistiksel olarak anlamlı bir farklılık yoktur." + color.END)
    else:
        print(color.BOLD + color.DARKCYAN + "Lütfen q değerini 3-4-5-6 sayılarından seçiniz.." + color.END)


ornek = pd.Series([5, 2] * 50)
isqcut_ok(rfm["recency"], 3)
isqcut_ok(rfm["frequency"], 5)
isqcut_ok(rfm["monetary"], 5)
isqcut_ok(ornek, 2)






rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması


# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


temp_df.loc[temp_df["group"] == 2, "score"].hist()

plt.title("grouped Sütunu Histogramı")
plt.xlabel("Değerler")
plt.ylabel("Sıklık")
plt.show(block=True)


liste1 = [16, 19, 20, 21, 22, 23, 24, 25, 26, 14,
         24, 25, 21, 18, 15, 24, 20, 18, 17, 24,
         36, 38, 41, 24, 29, 51, 37, 39, 41, 44,
         58, 55, 59, 47, 61, 64, 39, 62, 45, 53,
         75, 73, 67, 46, 81, 88, 67, 75, 79, 83]

series1 = pd.Series(liste1)

liste2 = [9, 11, 10, 21, 12, 13, 8, 15, 11, 7,
         24, 25, 21, 18, 15, 24, 20, 18, 17, 24,
         36, 48, 41, 24, 29, 51, 37, 39, 41, 44,
         58, 55, 59, 55, 61, 64, 39, 62, 60, 53,
         75, 73, 81, 61, 81, 88, 67, 75, 87, 83]

series2 = pd.Series(liste2)

isqcut_ok(series2, 5)
isqcut_ok(series1, 4)

