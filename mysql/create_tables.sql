CREATE TABLE vitals (
    hadm_id INT NOT NULL,
    charttime DATETIME NOT NULL,
    HeartRate DECIMAL(10, 1),
    SysBP DECIMAL(10, 1),
    DiasBP DECIMAL(10, 1),
    MeanBP DECIMAL(10, 1),
    RespRate DECIMAL(10, 1),
    TempC DECIMAL(20,16),
    Sp02 DECIMAL(10, 1),
    PRIMARY KEY (hadm_id, charttime)
);

LOAD DATA LOCAL INFILE '/tmp/processed_pivoted_vitals.csv'  INTO TABLE vitals  FIELDS TERMINATED BY ','  ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

CREATE TABLE gcs (
    hadm_id INT NOT NULL,
    charttime DATETIME NOT NULL,
    Sp02 INT,
    PRIMARY KEY (hadm_id, charttime)
);

LOAD DATA LOCAL INFILE '/tmp/processed_pivoted_gcs.csv'  INTO TABLE gcs  FIELDS TERMINATED BY ','  ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

CREATE TABLE labs (
    hadm_id INT NOT NULL,
    charttime DATETIME NOT NULL,
    SODIUM DECIMAL(10, 1),
    CHLORIDE DECIMAL(10, 1),
    GLUCOSE DECIMAL(10, 1),
    BUN DECIMAL(10, 1),
    CREATININE DECIMAL(10, 1),
    WBC DECIMAL(10, 1),
    BANDS DECIMAL(10, 1),
    HEMOGLOBIN DECIMAL(10, 1),
    HEMATOCRIT DECIMAL(10, 1),
    ANIONGAP DECIMAL(10, 1),
    PLATELET DECIMAL(10, 1),
    PTT DECIMAL(10, 1),
    PT DECIMAL(10, 1),
    INR DECIMAL(10, 1),
    BICARBONATE DECIMAL(10, 1),
    LACTATE DECIMAL(10, 1),
    PRIMARY KEY (hadm_id, charttime)
);

LOAD DATA LOCAL INFILE '/tmp/processed_pivoted_labs.csv'  INTO TABLE labs  FIELDS TERMINATED BY ','  ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

CREATE TABLE urine_output (
    hadm_id INT NOT NULL,
    charttime DATETIME NOT NULL,
    value DECIMAL(10, 1),
    PRIMARY KEY (hadm_id, charttime)
);

LOAD DATA LOCAL INFILE '/tmp/processed_urine_output.csv'  INTO TABLE urine_output  FIELDS TERMINATED BY ','  ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;