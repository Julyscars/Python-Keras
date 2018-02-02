"""
import xlrd


class ExcelUtil:
    def __init__(self, excel_path, sheet_name):
        self.data = xlrd.open_workbook(excel_path)
        self.table = self.data.sheet_by_name(sheet_name)
        # 获取第一行作为key值
        self.keys = self.table.row_values(0)
        # 获取总行数
        self.rowNum = self.table.nrows
        # 获取总列数
        self.colNum = self.table.ncols

    def dict_data(self):
        if self.rowNum <= 1:
            print("总行数小于1")
        else:
            r = []
            j = 1
            for i in range(self.rowNum - 1):
                s = {}
                # 从第二行取对应values值
                values = self.table.row_values(j)
                for x in range(self.colNum):
                    s[self.keys[x]] = values[x]
                r.append(s)
                j += 1
            return r


if __name__ == "__main__":
    filePath = "d:/2017年1月到12月物业服务管理工单清单.xlsx"
    sheetName = "2017年物业服务管理工单清单"
    data = ExcelUtil(filePath, sheetName)
    print(data.dict_data())
"""
import pandas as pd
data_xls = pd.read_excel('d:/2017年1月到12月物业服务管理工单清单.xlsx', '2017年物业服务管理工单清单', index_col = 0)
data_xls.to_csv('d:/2017年物业服务管理工单清单.csv', encoding='utf-8')
df=pd.read_csv('d:/2017年物业服务管理工单清单.csv')

print(df.head())
