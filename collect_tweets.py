import urllib
import httplib
import json
import datetime, time
import sys

API_KEY = '09C43A9B270A470B8EB8F2946A9369F3'
host = 'api.topsy.com'
url = '/v2/content/tweets.json'


def collectTweets(keyword, start_date, end_date, time_step, log_file_name, file_name):

    #########   create UNIX timestamps
    logfile = open(log_file_name, "a+")
    
    #all_tweets = []
    # modifiy time
    cur_start_date = start_date
    cur_end_date = cur_start_date + datetime.timedelta(seconds=time_step)
    time_step_modifier = 1.0
    while (cur_end_date <= end_date and cur_start_date < cur_end_date):

        mintime = int(time.mktime(cur_start_date.timetuple()))
        maxtime = int(time.mktime(cur_end_date.timetuple()))
        #########   set query parameters
        params = urllib.urlencode({'apikey' : API_KEY, 'q' : keyword,
                           'mintime': str(mintime), 'maxtime': str(maxtime),
                           'new_only': '1', 'include_metrics':'1', 'limit': 500})

        #########   create and send HTTP request
        req_url = url + '?' + params
        req = httplib.HTTPConnection(host)
        req.putrequest("GET", req_url)
        req.putheader("Host", host)
        req.endheaders()
        req.send('')

        #########   get response and print out status
        resp = req.getresponse()
        #print resp.status, resp.reason

        #########   extract tweets
        resp_content = resp.read()
        ret = json.loads(resp_content)
        tweets = ret['response']['results']['list']
        if len(tweets) < 500:
            time_step_modifier = 1.0
            #all_tweets.append(tweets)
            writeTweetsInFile(tweets, file_name)

            # write into logfile
            line = keyword.ljust(15) + "\tFrom: " + str(cur_start_date) + "\tTo: " + str(cur_end_date) + "\tNo. of Results: " + str(len(tweets)) + "\n"
            logfile.write(line)
            print keyword, " From:", cur_start_date, " To:", cur_end_date, " No. of Results:", len(tweets)
            # next time slot
            cur_start_date = cur_end_date
            cur_end_date = cur_start_date + datetime.timedelta(seconds=time_step)
            if cur_end_date > end_date:
                cur_end_date = end_date
        else:
            time_step_modifier *= 2.0
            cur_end_date = cur_start_date + datetime.timedelta(seconds=time_step/time_step_modifier)

    logfile.close()        

    #return all_tweets

def writeTweetsInFile(tweets, file_name):
    fo = open(file_name, "a+")
    for tweet in tweets:
        textTweet = json.dumps(tweet['tweet']['text']) #extract only textual part of tweets
        textTweet = textTweet[1: len(textTweet)-1] #remove ""
        fo.write(textTweet + "\n")
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Give hashtags")
        exit()
    hashtags = [hashtag for hashtag in sys.argv[1:]]
    start_date = datetime.datetime(2015,03,15, 00,00,0)
    end_date = datetime.datetime(2015,05,15, 00,00,0)
    time_step = 10000 #sec
    log_file_name = "search_log.txt"
    file_name = hashtags[0] + "_" + start_date.strftime('%m%d%Y') + "-" + end_date.strftime('%m%d%Y') + "_tweets.txt"
    fo = open(file_name, "w")
    for hashtag in hashtags:
        collectTweets(hashtag, start_date, end_date, time_step, log_file_name, file_name)
    fo.close()




    
