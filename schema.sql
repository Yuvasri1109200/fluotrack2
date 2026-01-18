CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
);

CREATE TABLE samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    sample_id TEXT,
    location TEXT,
    operator TEXT,
    particle_count INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
