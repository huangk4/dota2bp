import pandas as pd
import numpy as np
import csv

def handledata():
    data=pd.read_csv(r"E:\code\AI\dota2\matches_list_ranking.csv")
    datalen=len(data)
    print(datalen)
    radiant_data=data.iloc[:,2:7]
    radiant_data=np.array(radiant_data)
    dire_data=data.iloc[:,7:12]
    dire_data=np.array(dire_data)
    heros=130*2

    j=0
    k=0
    with open(r"E:\code\AI\dota2\test_data.csv","w") as train_file:
        writer  = csv.writer(train_file)
        for i in range(datalen):
            hero_vector = np.zeros(heros)
            for radiant in radiant_data[i]:
                hero_vector[radiant]=1
            for dire in dire_data[i]:
                hero_vector[120+dire]=1
            writer.writerow(hero_vector)
            j+=1
            if j>10000:
                k+=1
                print(k)
                j=0


if __name__ == "__main__":
    handledata()





