import requests
import dota2api

def way1():
    url="https://api.opendota.com/api/heroes"
    text=requests.get(url)
    with open("heroes.txt",'w') as hero:
        hero.write(text.text)

def way2():
    api = dota2api.Initialise("EB913C00E4E04F71DF73DE8A05578B83", raw_mode=True,language="zh_CN")
    data=api.get_heroes()
    id_name={}
    for i in data['heroes']:
       id_name[i['id']]=i['localized_name']
    #print(id_name)
    with open("heroes.txt",'w') as hero:
        for id,name in id_name.items():
            hero.write(str(id)+' '+name+'\n')

if __name__ == "__main__":
    way2()