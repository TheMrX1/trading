import sqlite3
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "bot_database.db")
USERS_FILE = os.path.join(BASE_DIR, "users.txt")
BLACKLIST_FILE = os.path.join(BASE_DIR, "blacklist.txt")

logger.info(f"Base Directory: {BASE_DIR}")
logger.info(f"Looking for users file at: {USERS_FILE}")
logger.info(f"Looking for blacklist file at: {BLACKLIST_FILE}")
logger.info(f"Database will be created at: {DB_NAME}")

def create_tables(cursor):
    """Creates the necessary tables if they don't exist."""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ticker TEXT,
        comment TEXT,
        custom_name TEXT,
        UNIQUE(user_id, ticker),
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ticker TEXT,
        quantity INTEGER,
        avg_price REAL,
        UNIQUE(user_id, ticker),
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id INTEGER PRIMARY KEY,
        chart_type TEXT DEFAULT 'static',
        advises_interval TEXT DEFAULT '1M',
        extra_funds REAL DEFAULT 0.0,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS blacklist (
        ticker TEXT PRIMARY KEY,
        added_by_user_id INTEGER,
        comment TEXT,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

def parse_users_file(filepath):
    """Parses the legacy users.txt file."""
    if not os.path.exists(filepath):
        logger.warning(f"{filepath} not found. Skipping user data migration.")
        return {}

    data = {}
    current_user_id = None
    current_section = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("USER_ID:"):
            current_user_id = int(line.split(":")[1])
            data[current_user_id] = {
                "assets": [],
                "comments": {},
                "names": {},
                "portfolio": {},
                "settings": {},
                "extra_funds": 0.0
            }
            current_section = None
        elif line == "ASSETS:":
            current_section = "ASSETS"
        elif line == "END_ASSETS":
            current_section = None
        elif line == "COMMENTS:":
            current_section = "COMMENTS"
        elif line == "END_COMMENTS":
            current_section = None
        elif line == "ASSET_NAMES:":
            current_section = "ASSET_NAMES"
        elif line == "END_ASSET_NAMES":
            current_section = None
        elif line == "PORTFOLIO:":
            current_section = "PORTFOLIO"
        elif line == "END_PORTFOLIO":
            current_section = None
        elif line == "SETTINGS:":
            current_section = "SETTINGS"
        elif line == "END_SETTINGS":
            current_section = None
        elif line == "EXTRA_FUNDS:":
            current_section = "EXTRA_FUNDS"
        elif line == "END_EXTRA_FUNDS":
            current_section = None
        elif current_user_id is not None:
            if current_section == "ASSETS":
                data[current_user_id]["assets"].append(line)
            elif current_section == "COMMENTS":
                if "=" in line:
                    parts = line.split("=", 1)
                    data[current_user_id]["comments"][parts[0]] = parts[1]
            elif current_section == "ASSET_NAMES":
                if "=" in line:
                    parts = line.split("=", 1)
                    data[current_user_id]["names"][parts[0]] = parts[1]
            elif current_section == "PORTFOLIO":
                if "=" in line:
                    parts = line.split("=", 1)
                    ticker = parts[0]
                    vals = parts[1].split(",")
                    if len(vals) == 2:
                        data[current_user_id]["portfolio"][ticker] = {
                            "qty": int(vals[0]),
                            "avg_price": float(vals[1])
                        }
            elif current_section == "SETTINGS":
                if "=" in line:
                    parts = line.split("=", 1)
                    data[current_user_id]["settings"][parts[0]] = parts[1]
            elif current_section == "EXTRA_FUNDS":
                try:
                    data[current_user_id]["extra_funds"] = float(line)
                except:
                    pass
                    
    return data

def parse_blacklist_file(filepath):
    """Parses the legacy blacklist.txt file."""
    if not os.path.exists(filepath):
        logger.warning(f"{filepath} not found. Skipping blacklist migration.")
        return {}
        
    blacklist_data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        if "=" in line:
            parts = line.split("=", 2)
            if len(parts) >= 3:
                ticker = parts[0]
                user_id = int(parts[1])
                comment = parts[2]
                blacklist_data[ticker] = {"user_id": user_id, "comment": comment}
    return blacklist_data

def migrate_data():
    """Main migration function."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    logger.info("Creating tables...")
    create_tables(cursor)
    
    # Migrate Users Data
    users_data = parse_users_file(USERS_FILE)
    logger.info(f"Found {len(users_data)} users to migrate.")
    
    for user_id, data in users_data.items():
        # 1. Insert User
        cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        
        # 2. Insert Assets
        for ticker in data["assets"]:
            comment = data["comments"].get(ticker, "")
            custom_name = data["names"].get(ticker, "")
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO assets (user_id, ticker, comment, custom_name) VALUES (?, ?, ?, ?)",
                    (user_id, ticker, comment, custom_name)
                )
            except Exception as e:
                logger.error(f"Error inserting asset {ticker} for user {user_id}: {e}")
        
        # 3. Insert Portfolio
        for ticker, pos in data["portfolio"].items():
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO portfolio (user_id, ticker, quantity, avg_price) VALUES (?, ?, ?, ?)",
                    (user_id, ticker, pos["qty"], pos["avg_price"])
                )
            except Exception as e:
                logger.error(f"Error inserting portfolio {ticker} for user {user_id}: {e}")
                
        # 4. Insert Settings
        settings = data["settings"]
        chart_type = settings.get("chart_type", "static")
        advises_interval = settings.get("advises_interval", "1M")
        extra_funds = data["extra_funds"]
        
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO user_settings (user_id, chart_type, advises_interval, extra_funds) VALUES (?, ?, ?, ?)",
                (user_id, chart_type, advises_interval, extra_funds)
            )
        except Exception as e:
            logger.error(f"Error inserting settings for user {user_id}: {e}")

    # Migrate Blacklist
    blacklist_data = parse_blacklist_file(BLACKLIST_FILE)
    logger.info(f"Found {len(blacklist_data)} blacklist entries to migrate.")
    
    for ticker, data in blacklist_data.items():
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO blacklist (ticker, added_by_user_id, comment) VALUES (?, ?, ?)",
                (ticker, data["user_id"], data["comment"])
            )
        except Exception as e:
            logger.error(f"Error inserting blacklist {ticker}: {e}")

    conn.commit()
    conn.close()
    logger.info("Migration completed successfully.")

if __name__ == "__main__":
    migrate_data()
