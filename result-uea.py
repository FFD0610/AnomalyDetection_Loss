#load the result from best-result file from each fold experiments
import numpy as np
import pandas as pd
import os
import xlsxwriter
import statistics
import pandas.io.formats.excel

#index = flag(class)
def best_inform_matrix(matrix_dir,index):
    arrs = None
    with open(matrix_dir,'r') as f:
        lines = f.readlines()
        end_idx = len(lines)
        start_idx = [i for i,_ in enumerate(lines) if '[[' in _][0]
        #start_idx = len(lines) - len(index)
        matrix_lines = lines[start_idx:end_idx]
        #print(matrix_lines)
        for i,matrix_line in enumerate(matrix_lines):

            if ']' not in matrix_line:
                line = matrix_line.strip('\n')
                line = line + matrix_lines[i+1]
                line = line.strip('\n')
            else:
                line = matrix_line.strip('\n')


            if '[' not in matrix_line:
                continue


            line = line.replace('[','')

            line = line.replace(']','')

            #print(i, line, matrix_line)
            
            line = list(map(int,line.split()[:len(index)]))
            line = np.array(line)
            
            if arrs is None:
                arrs = line
            else:
                arrs = np.vstack((arrs,line))
    arrs = arrs[:len(index)]
    #print('length ' + str(arrs.shape[0]) + ' ' + str(len(index)))

    assert arrs.shape[0] == len(index)

    return arrs



def each_class(matrix,fault_flags):
    f1s, pres, recs, accs = [], [], [], []
    batch_sum = np.sum(matrix.flatten())

    for fault_flag in fault_flags:
        #index = fault_flags.index(fault_flag)
        index = fault_flag
        TP = matrix[index,index]
        
        FP_TP = np.sum(matrix[:,index],axis=0)
        TP_FN = np.sum(matrix[index],axis=0)
        TN = batch_sum + TP - FP_TP - TP_FN 
        if FP_TP == 0:
            FP_TP = 1e10
        if TP_FN == 0:
            TP_FN = 1e10
        pre = float(TP)/FP_TP
        rec = float(TP)/TP_FN
        acc = float(TP+TN)/batch_sum
        if pre + rec == 0:
            f1 = 1/float(1e10) 
        else:
            f1 = 2*pre*rec/(pre+rec)

        f1s.append(f1)
        pres.append(pre)
        recs.append(rec)
        accs.append(acc)

    f1 = sum(f1s)/len(f1s)
    pre = sum(pres)/len(f1s)
    rec = sum(recs)/len(recs)
    acc = sum(accs)/len(accs)
    return f1, pre, rec, acc

def load_f1(file_path):
    f1s, pres, recs, accs = [], [], [], []
    with open(file_path,'r') as f:
        lines = f.readlines()
        end_idx = [i for i,_ in enumerate(lines) if '[[' in _][0] -1 
        start_idx = [i for i,_ in enumerate(lines) if 'class 0' in _][0]
        #start_idx = len(lines) - len(index)
        matrix_lines = lines[start_idx:end_idx]
        #print(matrix_lines)
        for i,matrix_line in enumerate(matrix_lines):
            #for each class 
            #class 1: acc = 0.9630480167014613, precision = 0.2834285714285714, recall = 0.28837209302325584, F1 score = 0.2858789625360231
            matrix_line = matrix_line.split('acc = ')
            acc, matrix_line = matrix_line[-1].split(', precision =')
            #print(matrix_line, matrix_line[-1].split(', recall = '))
            pre, matrix_line = matrix_line.split(', recall = ')
            rec, matrix_line = matrix_line.split(', F1 score =')
            f1 = matrix_line.strip('\n').strip(', F1 score =')
        f1s.append(float(f1))
        pres.append(float(pre))
        recs.append(float(rec))
        accs.append(float(acc))

    f1 = sum(f1s)/len(f1s)
    pre = sum(pres)/len(f1s)
    rec = sum(recs)/len(recs)
    acc = sum(accs)/len(accs)
    return f1, pre, rec, acc

if __name__ == '__main__':

    flag_index  = ['12_FLW4_1','29_VLVBPS', '30_VLVACL', '31_GCI', '32_34_GC1P1_3', '35_GC2P', '38_WPMP1', '41_VVPWC1', '42_VVPWC2']#,'Normal']
    loss_name_list = ['center-zerograd','soft-contra', 'design-maskedonlyforcls-.7','softmax','hard-contra','mixed','softmax-re', 'design-maskedonlyforcls-.7-real',]
    index = ['Data_name'] + loss_name_list + ['Num-class', 'Data-dim', 'Lenght', 'Train-size', 'Test-size']
    #TrainSize	TestSize	NumDimensions	SeriesLength

    #index1 = ['time_steps', 'loss', 'model', 'F1', 'k', 'epoch']
    result_dir = '/mnt/database/torch_results/tanaka-try/uea-re' #'./log/codetesting' #result path you want to load 
    if '10' in result_dir:
        flag_index.append('Normal')
    output_dir =   './excel/uea' #excel file output path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #df_f1 = pd.DataFrame(columns=index)
    df_f1, df_pre, df_rec, df_acc = [ pd.DataFrame(columns=index) for i in range(4) ]
    df_list = [df_f1, df_pre, df_rec, df_acc]
    xsl_name = 'uea_250219'
    writer = pd.ExcelWriter(os.path.join(output_dir,f'{xsl_name}.xlsx')) 
    result_all = {}
    #for time_steps in os.listdir(result_dir):
    #for time_steps in ['100','200']: ####################
    data_info = pd.read_csv('/mnt/database/datasets/UEA/DataDimensions.csv',usecols=range(10))
    print(data_info.columns, data_info.index)
    columns = data_info.columns.tolist()
    for data_name in os.listdir(os.path.join(result_dir)):#['50']: ####################
        result_all[data_name] = {}
        flag_index = np.arange(data_info.loc[data_info['Problem']==data_name,'NumClasses'].values)
        dim_data = list(data_info.loc[data_info['Problem']==data_name,['NumDimensions', 'SeriesLength', 'TrainSize', 'TestSize']].values.flatten()) #
        print(dim_data)
        for loss_name in loss_name_list:#os.listdir(os.path.join(result_dir, data_name)):
            result_all[data_name][loss_name] = {}

            for model_name in ['tcn']:#os.listdir(os.path.join(result_dir, data_name, loss_name)):
                result_all[data_name][loss_name][model_name] = [] #f1 pre rec acc for each
            
                file_path = os.path.join(result_dir, data_name, loss_name, model_name, 'best_results.txt')
                if not os.path.exists(file_path):
                    #not completed training
                    result_all[data_name][loss_name][model_name].extend([-1,-1,-1,-1, len(flag_index)])
                    continue
                print(file_path)
                with open(file_path, 'r') as file:
                    epoch = int(file.readlines()[0].strip('\n').split(':')[-1] )
                if  data_name == 'Phoneme':
                #if (loss_name == 'mixed' or loss_name == 'softmax-pooling') and (model == 'timesnet' or model == 'dlinear'):
                    #continue
                    f1, pre, rec, acc = load_f1(file_path)

                else:
                    k_matrix = best_inform_matrix(file_path, flag_index)
                    print(k_matrix)
                    f1, pre, rec, acc = each_class(k_matrix, flag_index)
                result_all[data_name][loss_name][model_name].extend([f1, pre, rec, acc, len(flag_index)])

        for i, _ in enumerate(df_list):
            values = [ result_all[data_name][loss_name]['tcn'][i] for loss_name in loss_name_list]
            result = [data_name] + values + [result_all[data_name][loss_name]['tcn'][-1] ] + dim_data
            _.loc[len(_.index)] = result


    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd.io.formats.excel.header_style = None
    sheet_names = ['f1', 'pre', 'rec', 'acc']
    for i, _ in enumerate(sheet_names):
        df_list[i].to_excel(writer,sheet_name=sheet_names[i])


        workbook = writer.book
        worksheets = writer.sheets
        worksheet = worksheets[sheet_names[i]]

        fmt = workbook.add_format({'font_name': 'Arial','align': 'center'})
        worksheet.set_column('A:C', 25, fmt)
        worksheet.set_column('D:N', 25, fmt)
        format1 = workbook.add_format({'bold': 1})
        format2 = workbook.add_format({'num_format': '%.4f'})
        format3 = workbook.add_format({'left':6})

    writer.close()
    df_list[0].to_csv(os.path.join(output_dir,'{xls_name}_f1.csv'),index=False)
    #df.to_csv(os.path.join(output_dir,'design_10cls.csv'),index=False)
    #df.to_csv(os.path.join(output_dir,'softmax-aug.csv'),index=False) ################