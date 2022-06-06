import math
from typing import *
from xmlrpc.client import DateTime
import numpy.random as npr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import db # Model that handles the SQL connections
from tqdm import tqdm

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



class Question:
    """Class for questions that users can answer.

    A question object that used to model a question

    Attributes:
      p:
        A float between 0 and 1, the chance of getting right by guessing.
      qid:
          An integer that used for identifying this question.
    """

    def __init__(self, p: float, qid: int):
        """
        Initialize a Question object.

        Args:
          p: A float between 0 and 1, the chance of getting right by guessing.
          qid: An integer, the ID number of the question.

        Raise:
          ValueError: An error is raise if p is not in [0,1].
        """
        self._p = p
        self._qid = qid
        if p < 0 or p > 1:
            raise ValueError(f"The probability {p} is not between 0 and 1")
        conn = db.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO questions
                    (`qid`)
                    VALUES({qid});
                """
            )
            conn.commit()

    @property
    def p(self):
        """
        A float between 0 and 1, the chance of getting right by guessing."""
        return self._p

    @p.setter
    def p(self, p):
        """
        A float between 0 and 1, the chance of getting right by guessing.
        """
        self._p = p

    @property
    def qid(self):
        """The ID of the question"""
        return self._qid


class User:
    """
    Class for modling users that can answer questions and learn.

    A user object can generate answers and ratings when a question object given.

    Attributes:
      uid:
        An integer that used for identifying this user.
    """

    def __init__(self, uid: int):
        """
        Initialize a user object.

        Args:
          uid: An integer that used for identifying this user
        """
        self._uid = uid

        conn = db.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO users
                    (`uid`)
                    VALUES({uid});
                """
            )
            conn.commit()

    @property
    def uid(self):
        """An integer that used for identifying this user"""
        return self._uid

    @uid.setter
    def uid(self, uid):
        """An integer that used for identifying this user."""
        conn = db.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
            UPDATE users
                SET `uid` = {uid}
                WHERE `uid` = {self.uid};
                """
            )
            conn.commit()
        self._uid = uid

    def _generate_correct(self, question: Question, **kwargs) -> bool:
        """
        Internal method used to generate whether the user answered the question correctly.

        Args:

          question:
            A question object for which response will be generated for.
          repetition:
            An integer,the repetition number of this user for this question prior this interaction.

        Returns:

            correct:
              A boolean indicating whether the user answer `question` correctly.
        """
        # returns True with probability p
        corerect = npr.rand() < question.p
        return corerect

    def _generate_difficulty(self, **kwargs) -> int:
        """
        Internal method used to generate difficulty ratings.

        Args:

          question:
            A question object for which difficulty rating will be generated for .

        Returns:

            difficulty rating:
              An integer, an element possible difficulty ratings.
        """
        # randomly returns a answer between MIN_DIFFICULTY and MAX_DIFFICULTY
        return npr.randint(MIN_DIFFICULTY, MAX_DIFFICULTY)

    def _generate_relevance(self, question, **kwargs) -> int:
        """
        Internal method used to generate relevance ratings.

        Args:

          question:
            A question object for which relevance rating will be generated for .

        Returns:

            relevance rating:
              An integer, an element possible relevance ratings.
        """
        # randomly returns a answer between MIN_RELEVANCE and MAX_RELEVANCE
        return npr.randint(MIN_RELEVANCE, MAX_RELEVANCE + 1)

    def repetitions(self, question: Union[Question, int]) -> int:
        """Take a question, returns the number of times this user interacted with this question.

        Args:
        
          quesion:
            A question object or a qid of a question object.

        Returns:
        
          repetition:
            An integer, number of times this user interacted with this question.
        """
        if isinstance(question, Question):
            qid = question.qid
        elif isinstance(question, int):
            qid = question

        conn = db.get_connection()
        with conn.cursor(buffered=True, dictionary=True) as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*) AS repetition
                FROM interactions WHERE
                `uid` = {self.uid} AND `qid` = {qid};
                """
            )
            rep = cursor.fetchall()
        return rep[0]["repetition"]

    def attempt(self, question:Question, day:datetime, **kwargs) -> dict:
        """
        Take a question, return a response to the question.

        Args:

          question: A question object.
          day: A Datetime.date object On which day this question is attempted

        Returns

           {q, u,correct, difficulty, relevance, question}: A dictionary, where `correct` is whether the student get the question right`,
           integer `difficulty` and `relevance` are the difficulty and relevance rating generated for the `question`.
        """
        interact_data = {"question": question, "day": day}
        correct = self._generate_correct(**interact_data)
        difficulty = self._generate_difficulty(**interact_data)
        relevance = self._generate_relevance(**interact_data)
        if not correct:
            difficulty = ERROR_DIFFICULTY
        return {
            "qid": question.qid,
            "uid": self.uid,
            "correct": correct,
            "difficulty": difficulty,
            "relevance": relevance,
        }

    def recent_correction_rate(self, qid:int, n=MATURE_SAMPLE_SIZE) -> float:
        """
        Calculate the average correction rate of last `n` times answering question with qid `qid`.
        """
        conn = db.get_connection()
        with conn.cursor(buffered=True) as cursor:
            cursor.execute(
                f"""
                      SELECT is_correct AS performance FROM interactions  
                      WHERE `uid` = {self.uid} AND `qid` = {qid} 
                      ORDER BY interact_time DESC LIMIT {n};
                      """
            )
        return np.mean(cursor.fetchall()[:n])


class Learner(User):
    """'
    The inheritance class of `User` that model user that will learn after repeated attempting
    a question.
    """

    def __init__(self, uid: int, learn_rate=16):
        """
        Create a Learner class that model a user that will learn by attempting a questions multiple
        times.

        Args:
        
          learn_rate:
            A float that control how repetition improves the correction rate, the higher
            the number, the more faster the correction rate improves. Deafult set to 16.
        """
        super().__init__(uid)
        self._learn_rate = learn_rate

    @property
    def learn_rate(self):
        return self._learn_rate

    @learn_rate.setter
    def learn_rate(self, learn_rate):
        self._learn_rate = learn_rate

    def _corret_pro(self, p:float, rep:int, n_day:int) -> float:
        """
        Calculate the the probability of correctly answering a question.

        Args:
          p:
            Float, the probability of answering the question by guessing
          rep:
            How many times has this question being seen
          n_day:
            How may day since last time the question been studied
        """
        return p + (1 - p) * np.exp(-1 * n_day / self.learn_rate * rep)

    def _generate_correct(self, question: Question, day:datetime) -> bool:
        """
            Internal method used to generate whether the user answered the question correctly.
            
        Args:
          question:
            A question object, the one user try to answer
          day:
            On which day the question is being asked
        
        Returns:
            A bool variabel indicating whether a correct response will be given by the user.
        """
        p = question.p  # getting correct by guessing
        rep = self.repetitions(question.qid)
        if rep == 0:
            return npr.uniform() < p
        else:
            conn = db.get_connection()
            with conn.cursor(buffered=True) as cursor:
                cursor.execute(
                    f"""
                            SELECT interact_time AS last_seen FROM interactions WHERE `uid` = {self.uid} AND `qid` = {question.qid} 
                            ORDER BY interact_time DESC LIMIT 1
                            """
                )
                last_seen = cursor.fetchall()[0][0]
            n_day = (day.date() - last_seen).days
            if n_day <= 0:
                return npr.uniform() < p
            else:
                return npr.uniform() < self._corret_pro(p, rep, n_day)


def write_log(interactions: list[dict]) -> None:
    """
    Update the database with data of interactions between user and questions.

    Args:
    
      A list of dictionaries where each dictionary contains the following elements:
        `q`: 
          int, `qid` of the question that being answere by the user
        `u`: 
          int, uid of the user who answered the question
        `corret`: 
          bool, whethere a correct answer is received.
        `difficulty`: 
          int, the difficulty rating submitted by the user.
        `relevance`: 
          int, the relevance rating submitted by the user.
        `day`: 
          Date, the date this interaction happens

    Returns:
        None
    """
    conn = db.get_connection()
    records = [
        (
            d["u"],
            d["q"],
            d["relevance"],
            d["correct"],
            d["difficulty"],
            d["day"].strftime("%Y-%m-%d"),
        )
        for d in interactions
    ]

    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO interactions
                (`uid`, `qid`, relevance, is_correct, difficulty, interact_time)
                VALUES(%s, %s, %s, %s, %s, %s);
            """,
            records,
        )
        conn.commit()


def spaced_repetition(
    correct: bool,
    difficulty: int,
    repetition: int,
    review_interval: float,
    ease_factor: float,
    **kwargs,
) -> dict:
    """
    The main spaced repetition algorithm that used to determine when the user should review a question next time.

    Args:

      correct:
        A bool variable, Did user answer question this time correctly?
      repetition:
        An integer, the repetition number of this question for the user.
      review_interval:
        An integer, the previous review interval.
      ease_factor:
        An float, the larger the number, the easier the question is

    Returns:

      Returns a dictionary containing following elements:
      
      review_interval:
        An integer, when it's the next review time.
      repetition:
        A float, the updated repetition number.
      ease_factor:
        A float, the updated ease factor.

    Note:
      The `repetition` values in spaced repetition algorithm is different from number of logged interactions between the `question` and the `user` as it can
      be set to 0 by the algorithm.
    """
    if correct:
        if repetition == 0:
            review_interval = 1
        elif repetition == 1:
            review_interval = 6
        elif review_interval > 1:
            review_interval = math.ceil(ease_factor * review_interval)
        ease_factor = max(
            1.3, ease_factor + 0.5 - 0.08 * difficulty - 0.02 * difficulty**2
        )
        repetition += 1
    else:
        repetition = 0
        review_interval = 1
    return {
        "review_interval": review_interval,
        "repetition": repetition,
        "ease_factor": ease_factor,
    }


def simulate(lst_q:list[Question], lst_u:list[User], max_day: int, start_day: datetime) -> None:
    """
    Take an integer `max_day` and date  `start_day` to simulate interactions between users and questions starting at `start_day` for `max_day`s.

    Args:
    
      lst_q: 
        A list of questions
      lst_u: 
        A list of users
      max_day: 
        int, maximum number of days to be simulated
      start_day: 
        daytime object, the date the simulation starts
    
    Returns: None

    """
    conn = db.get_connection()
    u_map = {u.uid: u for u in lst_u}
    q_map = {q.qid: q for q in lst_q}
    days = pd.date_range(start=start_day, periods=max_day).tolist()

    # initializing
    q_u_start_data = [
        (u.uid, q.qid, start_day.strftime("%Y-%m-%d"), u.uid, q.qid)
        for u in lst_u
        for q in lst_q
    ]
    with conn.cursor() as cursor:
        cursor.executemany(
            f"""
                    INSERT INTO question_user(`uid`, `qid`,repetition,is_matured, ease_factor,review_interval, next_study_day)
                    SELECT %s,%s,0,false,{INI_EASE_FACTOR},{INI_REVIEW_INTERVAL}, %s FROM DUAL
                    WHERE NOT EXISTS( SELECT * FROM question_user WHERE uid = %s AND `qid` = %s LIMIT 1);              
                          """,
            q_u_start_data,
        )
        conn.commit()

    # Simulate interactions on each day
    for day in tqdm(days):
        # get info on interactions planned to happen on this day
        with conn.cursor(buffered=True, dictionary=True) as cursor:
            cursor.execute(
                f"""
                SELECT
                    `uid`,
                    `qid`,
                    repetition,
                    review_interval,
                    ease_factor
                FROM
                    question_user
                WHERE
                    next_study_day = "{day.strftime('%Y-%m-%d')} AND is_mature = FALSE"
                        """
            )
            study_tasks = cursor.fetchall()

        lst_interactions_updates = []
        lst_q_u_updates = []
        lst_maturity_updates = []

        for task in study_tasks:  # simulate interactions
            q = q_map[task["qid"]]
            u = u_map[task["uid"]]
            matured = (
                u.repetitions(q) >= MATURE_MIN_ATTEMPTS
                and u.recent_correction_rate(q.qid) >= MATURE_ACCEPTANCE
            )
            if matured:
                lst_maturity_updates.append((u.uid, q.qid))
            else:
                interact_data = dict(u.attempt(q, day), **task)
                (  # parameters generated by the spaced_repetition algorithm
                    new_interval,
                    new_repetition,
                    new_ease_factor,
                ) = spaced_repetition(**interact_data).values()
                lst_interactions_updates.append(
                    (
                        task["uid"],
                        task["qid"],
                        interact_data["relevance"],
                        bool(interact_data["correct"]),
                        interact_data["difficulty"],
                        day.strftime("%Y-%m-%d"),
                    )
                )
                lst_q_u_updates.append(
                    (
                        new_repetition,  # repetition
                        bool(matured),  # is_mature
                        new_ease_factor,  # ease_factor
                        new_interval,  # review_interval
                        (day + timedelta(days=new_interval)).strftime(
                            "%Y-%m-%d"
                        ),  # next_study_day
                        task["uid"],
                        task["qid"],
                    )
                )
                # update the database with data
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE
                    question_user
                SET
                    repetition = %s,
                    is_matured = %s,
                    ease_factor = %s,
                    review_interval = %s,
                    next_study_day = %s
                WHERE
                    `uid` = %s
                    AND `qid` = %s;
                            """,
                lst_q_u_updates,
            )
            conn.commit()
            cursor.executemany(
                """
                UPDATE
                    question_user
                SET
                    is_matured = 1
                WHERE
                    `uid` = %s
                    AND `qid` = %s
              """,
                lst_maturity_updates,
            )
            cursor.executemany(
                """
                            INSERT INTO interactions(`uid`,`qid`,relevance,is_correct,difficulty,interact_time)
                                VALUES(%s, %s,%s,%s,%s,%s)
                            """,
                lst_interactions_updates,
            )
            conn.commit()
        # check maturity updates on the simulations conducted on the last simulated day
        if day == days[-1]:
            final_maturity_updates = []
            for task in study_tasks:  # simulate interactions
                q = q_map[task["qid"]]
                u = u_map[task["uid"]]
                matured = (
                    u.repetitions(q) >= MATURE_MIN_ATTEMPTS
                    and u.recent_correction_rate(q.qid) >= MATURE_ACCEPTANCE
                )
                if matured:
                    final_maturity_updates.append((u.uid, q.qid))
            with conn.cursor() as cursor:
                conn.commit()
                cursor.executemany(
                    """
                  UPDATE question_user
                  SET is_matured = 1
                  WHERE `uid` = %s AND `qid` = %s
                  """,
                    final_maturity_updates,
                )
