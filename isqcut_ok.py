import numpy as np
import datetime as dt
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, f_oneway, kruskal
from statsmodels.stats.multicomp import MultiComparison

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
