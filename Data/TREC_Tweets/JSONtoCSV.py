import os
import json
import os
import csv

def tweets_JSON_CSV(path):
    for dirs, subdirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                #print(dirs+file)
                with open(dirs+file, 'r',encoding='utf-8') as input_file:
                    print(dirs+file)
                    csv_out = open(dirs+file+'.csv', mode='w') #opens csv file
                    writer = csv.writer(csv_out) #create the csv writer object
                    fields=['tweet_id','Text','TimeStamp','Event_Type','User Info', 'MetaData','Place']
                    writer.writerow(fields) #writes field
                    for line in input_file:
                        try:
                            tweet = json.loads(line)
                            AllProperties=tweet['allProperties']                            
                            Event_Name=tweet['topic']
                            writer.writerow([AllProperties['id'],AllProperties['text'],AllProperties['user.created_at'],Event_Name])                        
                        except:
                            continue


#Train Data
Training_Data_Path='E:/HIT/TREC-IS/Implementation/TREC_IS 2019 Code/TREC-IS19/Data/Tweets Data/Train Data/'
tweets_JSON_CSV(Training_Data_Path)
#Test Data
Test_Data_Path='E:/HIT/TREC-IS/Implementation/TREC_IS 2019 Code/TREC-IS19/Data/Tweets Data/Test Data/'
tweets_JSON_CSV(Test_Data_Path)
