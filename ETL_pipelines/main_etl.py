import os
import logging
from retry import retry
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

load_dotenv()

PRODUCT_CSV_FILE_PATH = os.getenv("PRODUCT_CSV_FILE_PATH")
BRAND_CSV_FILE_PATH = os.getenv("BRAND_CSV_FILE_PATH")

# Load Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

NODES = [
    "Product", "Brand"
]

@retry(tries=10, delay=10)
def set_uniquness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR 
    (n:{node}) REQUIRE n.id IS UNIQUE;"""

    _ = tx.run(query)


@retry(tries=10, delay=10)
def create_uniqueness_constraints():
    LOGGER.info("Connecting to Neo4j...")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    LOGGER.info("Creating uniqueness constraints...")

    with driver.session() as session:
        for node in NODES:
            session.execute_write(set_uniquness_constraints, node)


def process_product_brand_csv(
    product_file_path: str,
    brand_file_path: str,

) -> pd.DataFrame:
    """
    Reads and processes CSV files containing product and data, merging them into a single DataFrame.

    This function reads data from three CSV files (product and brand), merges the data based on the 
    'categoryID' and 'supplierID' columns, and cleans the resulting DataFrame by replacing missing values in specific 
    columns with 'Unknown'.

    :param product_file_path: The file path to the CSV file containing product data.
    :param category_file_path: The file path to the CSV file containing category data.
    :param supplier_file_path: The file path to the CSV file containing supplier data.
    :return: A pandas DataFrame containing the merged product, category, and supplier data.
    """

    try:
        LOGGER.info(f"Reading data from {product_file_path}")
        product_df = pd.read_csv(product_file_path)

        LOGGER.info(f"Reading data from {brand_file_path}")
        brand_df = pd.read_csv(brand_file_path)

        LOGGER.info("Merging product and brand data")
        product_brand_df = pd.merge(
            product_df, brand_df, on='s_no', how='left')

        LOGGER.info("Cleaning data, replacing NA values with Unknown")
        product_brand_df["mpn"] = product_brand_df["mpn"].replace({pd.NA: "Unknown"})
        product_brand_df["price"] = product_brand_df["price"].replace({pd.NA: "Unknown"})
        product_brand_df["sku"] = product_brand_df["sku"].replace({pd.NA: "Unknown"})
        return product_brand_df
    except Exception as e:
        LOGGER.error(f"Error reading CSV data: {e}")


def insert_data(tx, row):
    """
    Inserts product, category, and supplier data into a Neo4j graph database.

    This function creates a product node, merges category and supplier nodes, and 
    establishes relationships between the product and its category and supplier in 
    the Neo4j graph. The data is passed as a dictionary in the `row` parameter.

    :param tx: The transaction object used to execute the Cypher queries in the Neo4j database.
    :param row: A dictionary containing the product, category, and supplier data to be inserted.
    """

    tx.run('''
            CREATE (product:Product {
                s_no: $s_no,
                name: $name,
                sku: $sku,
                mpn: $mpn,
                price: $price,
                in_stock: $in_stock,
                currency: $currency,
                images: $images,
                gender: $gender
            })
           MERGE (brand:Brand {
                s_no: $s_no,
                brand_name: $brand_name,
                description: $description
            })
           CREATE (product)-[:MAKER]->(brand)
            ''', row)

@retry(tries=100, delay=10)
def load_product_brand_into_graph(
    product_brand_df: pd.DataFrame,
):
    """
    Loads product, category, and supplier data into a Neo4j graph database.

    This function connects to a Neo4j database and inserts data from a pandas DataFrame 
    that contains product, category, and supplier information. The data is inserted 
    into the graph using a write transaction. The function uses retry logic to ensure 
    successful data insertion, retrying up to 100 times with a 10-second delay between attempts.

    :param product_category_supplier_df: A pandas DataFrame containing the product, category, 
                                         and supplier data to be inserted into the graph.
    """

    LOGGER.info("Connecting to Neo4j")
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    LOGGER.info("Inserting data into Neo4j")
    with driver.session() as session:
        for _, row in product_brand_df.iterrows():
            session.execute_write(insert_data, row.to_dict())

    LOGGER.info("Data inserted into Neo4j")

    driver.close()

def main():
    LOGGER.info("Creating uniqueness constraints...")
    create_uniqueness_constraints()
    LOGGER.info("Uniqueness constraints created successfully.")

    LOGGER.info("Processing product and brand data...")
    product_brand_df = process_product_brand_csv(
        PRODUCT_CSV_FILE_PATH, BRAND_CSV_FILE_PATH
    )
    LOGGER.info("Data processed successfully.")

    LOGGER.info("Loading product and brand data into Neo4j...")
    load_product_brand_into_graph(product_brand_df)
    LOGGER.info("Data loaded successfully.")

if __name__ == "__main__":
    main()
