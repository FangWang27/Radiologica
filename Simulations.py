import importlib
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from SpaceRep_v2 import *
import SpaceRep_v2
importlib.reload(SpaceRep_v2)
from SpaceRep_v2 import *

conn = db.get_connection()
MIN_DIFFICULTY = 1 # Minimum difficulty a user can rate a question.
MAX_DIFFICULTY = 3 # Maximum difficulty a user can rate a question.
ERROR_DIFFICULTY = 4 # The difficulty rating of a user if user incorrely answer it.
INI_REVIEW_INTERVAL = 1 # The defult review_interval when a new question is encountered.
INI_EASE_FACTOR = 2.5 # The defult ease factor when a new question is encountered.
MIN_RELEVANCE = 1 # The minimum rating of 'relevance' a user can gave to a question.
MAX_RELEVANCE = 5 # The maximum rating of 'relevance' a user can gave to a question.
MATURE_MIN_ATTEMPTS = 5 # The minimum number of attempts user need to made to mature a question.
MATURE_ACCEPTANCE = 0.8 # The boundary value of the correction rate for a question to be matured.
MATURE_SAMPLE_SIZE = 5 # When calculate the "recent attempts", how many attempts should be included.




n_user = 60
n_q = 50
n_day = 100
lst_q = [Question(1 / 3, qid) for qid in range(1,n_q+1)]
lst_u = [Learner(uid) for uid in range(1,n_user+1)]
# create map to users from qid
u_map = {u.uid:u for u in lst_u}
q_map = {q.qid:q for q in lst_q}

#
start_day = datetime(2022, 5,23)
days = pd.date_range(start = start_day, periods=100).tolist()




simulate(lst_q, lst_u, n_day, start_day)
# Use User45 and June-1 as example 


conn = db.get_connection()

day = datetime(2022,6,1).strftime('%Y-%m-%d')
uid_1 = 45



# Obtain the performance of a "quiz" (defined to be questions attempted on June 1st) and population
# statistics for the quiz so a comparision with 'cohort' can be done.

#--------------------------------------------------------------------------------------------------
# performance against the cohort for a quiz
#--------------------------------------------------------------------------------------------------

with conn.cursor(buffered= True, dictionary= True) as cursor:
        cursor.execute(f'''
                       SELECT
    *
FROM
    (
        SELECT
            {uid_1} AS uid,
            SUM(c.correction_rate) AS mean,
            SUM((c.correction_rate) * (1 - c.correction_rate)) AS variance
        FROM
            (
                SELECT
                    qid,
                    AVG(is_correct) AS correction_rate
                FROM
                    interactions
                WHERE
                    interact_time <= "{day}"
                GROUP BY
                    qid
            ) AS c
        WHERE
            qid in(
                SELECT
                    qid
                FROM
                    interactions
                WHERE
                    uid = {uid_1}
                    AND interact_time = "{day}"
            )
    ) as pop_stats
    INNER JOIN (
        SELECT
            {uid_1} as uid,
            SUM(is_correct) as corrects,
            COUNT(*) as n
        FROM
            interactions
        WHERE
            uid = {uid_1}
            AND interact_time = "{day}"
    ) as obs_stats ON obs_stats.uid = pop_stats.uid;
                        ''')
        result = cursor.fetchall()[0]

# average correction rate
correction_rate =  float(result['corrects'] /result['n'])
z_score = float((result['corrects'] - result['mean'])/np.sqrt(result['variance']))
#plot the performance against cohort
x = np.linspace(-3, 3, 100)
plt.fill_between(x, 0, stats.norm.pdf(x), where=x <= z_score, alpha=0.3)
plt.plot(x, stats.norm.pdf(x))
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.show()


#--------------------------------------------------------------------------------------------------
# Maturity progress of the user
#--------------------------------------------------------------------------------------------------

with conn.cursor(buffered= True, dictionary= True) as cursor:
# Calculate the average performance of the most recent n (n = 5) times
    cursor.execute(f'''
CREATE OR REPLACE VIEW recent_avg AS
SELECT
    DISTINCT qid,
    AVG(is_correct) OVER(PARTITION BY qid) AS avg_r,
    n
FROM
    (
        SELECT
            qid,
            is_correct,
            ROW_NUMBER() OVER(
                PARTITION BY qid
                ORDER BY
                    interact_time DESC
            ) AS n
        FROM
            interactions
        WHERE
            uid = {uid_1}
            AND interact_time <= '{day}'
    ) as answer_record
WHERE
    answer_record.n <= {MATURE_SAMPLE_SIZE};
                   ''')
    conn.commit()
# get info on the average correction rate and as max(MATURE_SAMPLE_SIZE5, #attempts of the question)
    cursor.execute(f''' 
SELECT
   *
FROM recent_avg  
WHERE (qid,n) IN (
        SELECT qid, MAX(n) FROM recent_avg GROUP BY qid
    );
                   ''')
    maturity = pd.DataFrame(cursor.fetchall())
    

# Calculate maturity level; if n< MATURE_SAMPLE_SIZE, divide the maturity by half    

maturity['level_maturity'] = maturity.apply(lambda row:  min(1, float(row.avg_r)/MATURE_ACCEPTANCE) if row.n>= MATURE_SAMPLE_SIZE else  min(1, float(row.avg_r)/MATURE_ACCEPTANCE)/2, axis = 1)

    
plt.figure(figsize=(50, 1))
sns.heatmap(
    np.asarray(maturity['level_maturity']).reshape(1,n_q), linewidths=1, cmap="YlGn", linecolor="white", square=True
)
plt.show()

#--------------------------------------------------------------------------------------------------
# Maturity progress of the user comparision with the cohort
#--------------------------------------------------------------------------------------------------


# Bell curve on comparision of the maturity 

with conn.cursor(buffered= True,dictionary= True) as cursor:
    cursor.execute('''
SELECT
    uid,
    COUNT(*) as n_matured
FROM
    question_user
WHERE
    is_matured = TRUE
GROUP BY
    uid;
                   ''')
    maturities = pd.DataFrame(cursor.fetchall())
    
maturity_zscore = stats.zscore(maturities['n_matured'])
x = np.linspace(-4, 4, 200)
plt.fill_between(x, 0, stats.norm.pdf(x), where=x <= maturity_zscore[0], alpha=0.3)
plt.plot(x, stats.norm.pdf(x))
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.show()
