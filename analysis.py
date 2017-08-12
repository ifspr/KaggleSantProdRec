#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle


def cleanup(list_months,cc=0):   
  yyyy,mm,dd = list_months[0].split('-')
  m1="0"+str(int(mm)-1) if int(mm)-1 < 10 else str(int(mm)-1)
  date_m1 = yyyy+'-'+m1+'-'+dd
  list_months.append(date_m1)

  if (cc==1):
    yyyy,mm,dd = list_months[1].split('-')
    m1="0"+str(int(mm)-1) if int(mm)-1 < 10 else str(int(mm)-1)
    date_m1 = yyyy+'-'+m1+'-'+dd
    list_months.append(date_m1)
    
  outfile = "alldata_clean.csv"

  with open(outfile,"w") as outf:
    with open("train_ver2.csv") as f:
      print "Read train data...."
      header=f.readline().strip("\n").split(",")
      id_nomprov = header.index('"nomprov"')
      outf.write("%s\n" % ",".join([h.strip('"') for h in header]))  
      for line in f:
        cells = line.strip("\n").split(",")              
        if (cells[0] not in list_months): continue
        if len(cells) > len(header):            
           provname = cells[id_nomprov]+cells[id_nomprov+1]
           cells[id_nomprov] = provname         
           del cells[id_nomprov+1]
        cells = [cell.strip(" ") for cell in cells]   
        outf.write("%s\n" % ",".join(cells))
  
    with open("test_ver2.csv") as f:             
      print "Read test data...."
      header=f.readline().strip("\n").split(",")
      id_nomprov = header.index('"nomprov"')
      placeholder = "".join(["," for i in range(24)])
      for line in f:
        cells = line.strip("\n").split(",")              
        if (cells[0] not in list_months): continue
        if len(cells) > len(header):            
           provname = cells[id_nomprov]+cells[id_nomprov+1]
           cells[id_nomprov] = provname         
           del cells[id_nomprov+1]
        cells = [cell.strip(" ") for cell in cells]   
        outf.write("%s%s\n" % (",".join(cells),placeholder))
        

def filter_clients(df,products,months): 
  df_m15=df.loc[df.fecha_dato==months[2],['ncodpers']+list(products)]
  df_j15=df.loc[df.fecha_dato==months[0],:]
  
  dictname={}
  for i,prod in enumerate(products): dictname[prod]=prod+'_m15'
  df_m15.rename(columns=dictname,inplace=True)
  
  dictname={}
  for i,prod in enumerate(products): dictname[prod]=prod+'_j15'
  df_j15.rename(columns=dictname,inplace=True)
  
  df_mj15=pd.merge(df_m15,df_j15,how='right',on='ncodpers')
  del(df_m15)
  del(df_j15)
  
  for prod in products:
      prodm = prod+'_m15'
      df_mj15.loc[df_mj15[prodm].isnull(),prodm] = 0
  
  id= []
  Y = []
  for i,prod in enumerate(products):
      xt = df_mj15.loc[(df_mj15[prod+'_m15']==0) & (df_mj15[prod+'_j15']==1),'ncodpers']
      id.append(xt)
      Y.append(np.zeros(len(xt),dtype=np.int8)+i)
  
  idn = pd.concat(id)    
  Y = np.hstack(Y)
  
  mask=df_mj15.columns.isin(df_mj15.iloc[:1,].filter(regex="ind_+.*ult1_+"))
  features = df_mj15.columns[~mask].values   
  dff = df_mj15.loc[df_mj15.ncodpers.isin(idn.unique()),features]
  dffb = dff.append(df.loc[df.fecha_dato==fecha_test,features]).reset_index()
  return (idn,Y,dffb)


def av_pr_k(prediction,truth,k=7):
  if not len(truth): return 0
  predicted = prediction[:k]
  score=0.
  n=0.
  for i,pred in enumerate(predicted):
    if pred in truth and pred not in predicted[:i]:
       n+=1
       score+=n/(i+1.)
  return score/(min(k,len(truth)))

       
def mapk(lprediction,ltruth,k=7):
  return np.mean([av_pr_k(prediction,truth,k) for prediction,truth in zip(lprediction,ltruth)])


def transform_features(dff):
  ##-- drop non-feature properties 
  cols = ["ind_nuevo",
          "ult_fec_cli_1t",
          "conyuemp",
          "canal_entrada",
          "tipodom",
          "nomprov"]
  dff.drop(cols,axis=1,inplace=True)
  
  ##-- transform fecha_alta to 'alta_month' and 'alta_year'
  dff['alta_month']=dff.fecha_alta.map(lambda x:0.0 if type(x) is float else float(x.split("-")[1])).astype(np.int8)
  dff['alta_year']=dff.fecha_alta.map(lambda x:0.0 if type(x) is float else float(x.split("-")[0])).astype(np.int16)
  dff.drop("fecha_alta",axis=1,inplace=True)
  
  ##-- clean up & transform antiguedad  
  dff["antiguedad"] = dff["antiguedad"].map(lambda x: 0.0 if x < 0 or np.isnan(x) else x+1.0).astype(np.int16)
  
  ##-- replace nan in renta with median value
  dff.renta.fillna(np.nanmedian(dff.renta),inplace=True)
  
  ##-- replace entries in pais_residencia that is not ES to "NotES"
  dff.pais_residencia = dff.pais_residencia.map(lambda x:"ES" if x=="ES" else "NotES")
  
  ##-- convert non-nan entries of cod_prov to str
  dff.cod_prov=dff.cod_prov.map(lambda x:str(int(x)) if x>0.0 else x)
  
  ##-- shorten segmento to "T1","P2","U3"
  dff.segmento.replace({"03 - UNIVERSITARIO": "U3",
                        "02 - PARTICULARES": "P2",
                        "01 - TOP": "T1"},inplace=True)
  
  ##-- run pd.get_dummies
  onehot_cols = ["ind_empleado",
                 "pais_residencia",
                 "sexo",
                 "indrel",
                 "indrel_1mes", 
                 "tiprel_1mes",
                 "indresi",
                 "indext",
                 "indfall",
                 "cod_prov",
                 "ind_actividad_cliente",
                 "segmento"]
  return pd.get_dummies(dff, columns=onehot_cols) 
                

def prepare_xgb(dff2,idn,Y):  
  df_test = dff2.loc[dff2.fecha_dato==fecha_test,:].reset_index()
  df_test.drop(["level_0","index"],axis=1,inplace=True)
  df_tv = pd.DataFrame(idn)
  df_tv = pd.merge(df_tv,dff2.loc[dff2.fecha_dato == fecha_train,:],how="left",on="ncodpers")
  df_tv.drop("index",axis=1,inplace=True)
  mask = np.random.rand(len(df_tv)) < 0.8

  df_train = df_tv[mask].reset_index()
  df_train.drop("index",axis=1,inplace=True)
  df_validate = df_tv[~mask].reset_index()
  df_validate.drop("index",axis=1,inplace=True)
  
  Y_train = Y[mask]   
  Y_validate = Y[~mask]
  X_train = df_train.as_matrix()
  X_validate = df_validate.as_matrix()
  X_test = df_test.as_matrix()
  X_all = df_tv.as_matrix()
  Y_all = np.copy(Y)
  
  train = xgb.DMatrix(X_train[:,2:],label=Y_train,feature_names=df_train.columns.values[2:])
  validate = xgb.DMatrix(X_validate[:,2:],label=Y_validate,feature_names=df_validate.columns.values[2:])
  all_data = xgb.DMatrix(X_all[:,2:],label=Y_all,feature_names=df_tv.columns.values[2:])
  test = xgb.DMatrix(X_test[:,2:], feature_names=df_test.columns.values[2:])

  return (train,validate,all_data,test)


def train_xgb(train,val,all):  
  evallist  = [(train,'train'), (val,'eval')]
  param = {
          'objective': 'multi:softprob',
          'silent': 1,
          'eta': 0.1,
          'min_child_weight': 10,
          'max_depth': 8,
          'eval_metric': 'mlogloss',
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.9,
          'num_class': len(products)}
  model = xgb.train(param, train, 1000, evals=evallist, early_stopping_rounds=20)
  evallist  = [(comb,'all_data')]
  best_ntree_lim = int(model.best_ntree_limit)
  model = xgb.train(param, comb, best_ntree_lim, evals=evallist)
  outf = open("model.pkl", "wb")
  pickle.dump((model,best_ntree_lim),outf)
  outf.close()

  
def predict_xgb(test):
  
  model,best_ntree_lim = pickle.load(open('model.pkl','r'))
  return model.predict(test, ntree_limit=best_ntree_lim)


def make_list_7prod(prediction,products,len_test):   
  Y_test = []
  for i in range(len_test):
    y_test = [(pred,prod,iprod) for pred,prod,iprod in zip(prediction[i,:],products,range(len(products)))]
    y_test = sorted(y_test,key=lambda x:x[0],reverse=True)[:7]
    Y_test.append([ip for y,p,ip in y_test])   
  return Y_test


def create_ytrue(df,months,products):
  prodtest = df.loc[df.fecha_dato==months[1],['ncodpers']+[prod for prod in products]].reset_index()
  prodtest_mt1 = df.loc[df.fecha_dato==months[3],['ncodpers']+[prod for prod in products]].reset_index()
  dictname={}
  for i,prod in enumerate(products): dictname[prod]=prod+'_mmin1'
  prodtest_mt1.rename(columns=dictname,inplace=True)
  
  prodcomb=pd.merge(prodtest_mt1,prodtest,how='right',on='ncodpers')
  del(prodtest_mt1)
  del(prodtest)

  for prod in products:
    prodm = prod+'_mmin1'
    prodcomb.loc[prodcomb[prodm].isnull(),prodm] = 0
 
  id= []
  Y = []
  for i,prod in enumerate(products):
      xt = prodcomb.loc[(prodcomb[prod+'_mmin1']==0) & (prodcomb[prod]==1),'ncodpers']
      id.append(xt)
      Y.append(np.zeros(len(xt),dtype=np.int8)+i)
  
  idn = pd.concat(id)    
  Y = np.hstack(Y)
  idn_arr = idn.as_matrix()

  id_test = df.loc[df.fecha_dato==months[1],'ncodpers']

  Y_un=[]
  for i,id_un in enumerate(id_test):
    Y_un.append(Y[idn_arr == id_un])
  return Y_un


def actual_Y(row):
  l = []
  for i,prod in enumerate(products):
     if ((row[prod+'_mmin1'] == 0) & (row[prod] == 1)): l.append(i)
  return l   


if __name__ ==  "__main__":
  fecha_train = '2015-06-28'   #'2015-06-28'
  fecha_test = '2016-06-28'    #'2015-06-28'

  months = [fecha_train,fecha_test]
  cc=1 if (fecha_test != '2016-06-28') else 0
  cleanup(months,cc)

  dtypes = {"fecha_dato": str, "ncodpers": np.uint32}
  df = pd.read_csv("alldata_clean.csv", dtype=dtypes,low_memory=False)

  products = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
  (idn,Y,dff) = filter_clients(df,products,months)
  dff2 = transform_features(dff)
  (train,val,comb,test)=prepare_xgb(dff2,idn,Y)

  train_xgb(train,val,comb)
  prediction = predict_xgb(test)
  Y_test = make_list_7prod(prediction,products,len(dff2[dff2.fecha_dato==fecha_test]))
  outf = open("Ytest.pkl", "wb")
  pickle.dump((Y_test),outf)
  outf.close()

  if (cc==1):    
    Y_true = create_ytrue(df,months,products)
    map7 = mapk(Y_test,Y_true, 7)
    print map7  
      






      




















