import numpy as np
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
from datetime import date
import datetime
sns.set(color_codes=True)

STATE="Karnataka"

df_base_date = pd.read_csv("Stateanddistwisecount2.csv");
bappend:bool=False;

if(df_base_date[df_base_date["date"] == str(date.today())]. shape[0] == 0):
    bappend=True;

state=[];
district=[];
district_nm=[];
count=[];
todaysdate=[];
incr_percentage=[];

if(bappend):
    df_coviD_districWise=pd.read_json("https://api.covid19india.org/state_district_wise.json")  

    for item_state in df_coviD_districWise:
        
        for item_district in df_coviD_districWise[item_state]["districtData"]:
                state.append(item_state)
                district.append(item_district);
                tot_ln=len((item_district));
                dist_abbr=item_district[0]  +item_district[tot_ln-1];
                district_nm.append(dist_abbr);
                count.append(df_coviD_districWise[item_state]["districtData"][item_district]["confirmed"]);
                todaysdate.append(date.today());
                yesterday_count = df_base_date[(df_base_date["date"] == str(date.today() - datetime.timedelta(days=1))) &  (df_base_date["state"] == str(item_state))  & (df_base_date["district"] == str(item_district))];
                yesterday_count= yesterday_count.reset_index();
                if(len(yesterday_count["confirmed_count"].index) > 0 ):
                    yesterday_count_value= yesterday_count["confirmed_count"][0];
                    incr_percentage_today=(((df_coviD_districWise[item_state]["districtData"][item_district]["confirmed"]) - (yesterday_count_value))/yesterday_count_value)*100;
                else :
                    yesterday_count_value=0;

                incr_percentage.append(incr_percentage_today);

    zippedList =  list(zip( state,district, count,todaysdate,district_nm,incr_percentage))
    dfObj = pd.DataFrame(zippedList, columns = [ 'state','district','confirmed_count','date' , 'district_nm','incr_percentage'],index=None) 

if(bappend):
    dfObj.to_csv("Stateanddistwisecount2.csv",mode='a',index=None,header=False);
    dfObj = pd.concat([dfObj, df_base_date], ignore_index=True, copy=True)


fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
if(bappend):
    sns.scatterplot(x=dfObj[dfObj["state"]==STATE]["district"],y=dfObj[dfObj["state"]==STATE]["confirmed_count"],hue=dfObj[dfObj["state"]==STATE]["date"]);
    sns.pointplot(x=dfObj[dfObj["state"]==STATE]["district"],y=dfObj[dfObj["state"]==STATE]["confirmed_count"],hue=dfObj[dfObj["state"]==STATE]["date"]);
    plt.savefig(STATE + "_district_wise_daily_count"+ str(date.today())+".png")
else:
    sns.scatterplot(x=df_base_date[df_base_date["state"]==STATE]["district"],y=df_base_date[df_base_date["state"]==STATE]["confirmed_count"],hue=df_base_date[df_base_date["state"]==STATE]["date"]);
    sns.pointplot(x=df_base_date[df_base_date["state"]==STATE]["district"],y=df_base_date[df_base_date["state"]==STATE]["confirmed_count"],hue=df_base_date[df_base_date["state"]==STATE]["date"]);
    plt.savefig(STATE + "_district_wise_daily_count"+ str(date.today())+".png")

# plt.show()

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
if(bappend):
    sns.countplot(x="incr_percentage", hue="date", data=dfObj[dfObj["state"]==STATE]["district"])
    # sns.scatterplot(x=dfObj[dfObj["state"]==STATE]["district"],y=dfObj[dfObj["state"]==STATE]["incr_percentage"],hue=dfObj[dfObj["state"]==STATE]["date"]);
    plt.savefig(STATE + "_district_wise_daily_percent_count"+ str(date.today())+".png")
else:
    # sns.countplot(x="incr_percentage", hue="date", data=df_base_date[df_base_date["state"]==STATE]["district"])
    # sns.scatterplot(x=df_base_date[df_base_date["state"]==STATE]["district"],y=df_base_date[df_base_date["state"]==STATE]["incr_percentage"],hue=df_base_date[df_base_date["state"]==STATE]["date"]);
    plt.savefig(STATE + "_district_wise_daily_percent_count"+ str(date.today())+".png")

if(bappend):
    dfObj.to_json("district_wise_pan_india"+ str(date.today()) +".json");
else:
    df_base_date.to_json("district_wise_pan_india"+ str(date.today()) +".json");    



plt.show()