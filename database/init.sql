CREATE TABLE users(
    `uid` SMALLINT AUTO_INCREMENT PRIMARY KEY
);

CREATE TABLE questions(`qid` SMALLINT AUTO_INCREMENT PRIMARY KEY);

CREATE TABLE interactions(
    `interaction_id` INT UNSIGNED AUTO_INCREMENT,
    `uid` SMALLINT,
    `qid` SMALLINT,
    `relevance` TINYINT(5),
    `is_correct` BOOLEAN,
    `difficulty` TINYINT(5),
    `interact_time` DATE,
    CONSTRAINT pk_interaction PRIMARY KEY (`interaction_id`),
    CONSTRAINT fk_uid FOREIGN KEY (`uid`) REFERENCES users(`uid`),
    CONSTRAINT fk_qid FOREIGN KEY (`qid`) REFERENCES questions(`qid`)
);

CREATE TABLE question_user(
    `uid` SMALLINT,
    `qid` SMALLINT,
    `repetition` SMALLINT,
    `is_matured` BOOLEAN,
    `ease_factor` FLOAT(4),
    `review_interval` SMALLINT,
    `next_study_day` DATE,
    PRIMARY KEY (`uid`, `qid`),
    CONSTRAINT fk_question_user_uid FOREIGN KEY(`uid`) REFERENCES users(`uid`),
    CONSTRAINT fk_question_user_qid FOREIGN KEY(`qid`) REFERENCES questions(`qid`)
);

