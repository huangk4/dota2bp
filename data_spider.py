#coding:utf-8

import json
import requests
import time

base_url = 'https://api.opendota.com/api/publicMatches?less_than_match_id='
session = requests.Session()
session.headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
}

def crawl(input_url):
    crawl_tag = 3  #最多重试3次
    while crawl_tag>0:
        try:
            time.sleep(1)   # 暂停一秒，防止请求过快导致网站封禁。
            session.get("http://www.opendota.com/")  #获取了网站的cookie
            content = session.get(input_url)
            crawl_tag = 0
        except KeyboardInterrupt:
            exit(0)
        except:
            print(u"Poor internet connection. We'll have another try.")
            crawl_tag-=1
    json_content = json.loads(content.text)
    return json_content

max_match_id = 5096170501    # 设置一个极大值作为match_id，可以查出最近的比赛(即match_id最大的比赛)。
target_match_num = 500000  #要获取的数量
lowest_mmr = 3000  # 匹配定位线，筛选该分数段以上的天梯比赛 


recurrent_times = 0
data_count=0
with open('matches_list_ranking2.csv','w',encoding='gbk') as fout:
    fout.write('比赛ID, 时间, 天辉英雄,,,,, 夜魇英雄,,,,, 天辉是否胜利\n')
    while(data_count<target_match_num):
        match_list = []
        json_content = crawl(base_url+str(max_match_id))
        for i in range(len(json_content)):
            match_id = json_content[i]['match_id']
            radiant_win = json_content[i]['radiant_win']
            start_time = json_content[i]['start_time']
            avg_mmr = json_content[i]['avg_mmr']
            if avg_mmr==None:
                avg_mmr = 0
            lobby_type = json_content[i]['lobby_type']
            game_mode = json_content[i]['game_mode']
            radiant_team = json_content[i]['radiant_team']
            dire_team = json_content[i]['dire_team']
            duration = json_content[i]['duration']  # 比赛持续时间
            # if int(avg_mmr)<lowest_mmr:  # 忽略低分段比赛数据
            #     continue
            if int(duration)<900:   # 比赛时间过短，小于15min，视作有人掉线，忽略。
                continue
            if int(lobby_type)!=7 or (int(game_mode)!=3 and int(game_mode)!=22): #lobby_type=7为天梯,后两个为普通正常比赛
                continue
            x = time.localtime(int(start_time))
            game_time = time.strftime('%Y-%m-%d %H:%M:%S',x)
            one_game = [game_time,radiant_team,dire_team,radiant_win,match_id]
            match_list.append(one_game)
        max_match_id = json_content[-1]['match_id'] #获取更早的数据，避免数据重复
        recurrent_times += 1
        for i in range(len(match_list)):
            fout.write(str(match_list[i][4])+', '+match_list[i][0]+', '+match_list[i][1]+', '+\
                match_list[i][2]+', '+str(match_list[i][3])+'\n')
        data_count+=len(match_list)
        print(recurrent_times,data_count,max_match_id)