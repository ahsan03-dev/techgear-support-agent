import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Loading Hugging Face embedding model (Qwen/Qwen3-Embedding-0.6B)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

SUPPORT_ARTICLES = [
    {
        "category": "Returns & Refunds",
        "title": "30-Day Return and Refund Policy",
        "content": "TechGear offers a 30-day return policy for all hardware products. Items must be in their original packaging with all accessories included. Refunds are processed back to the original payment method within 5-7 business days after we receive the returned item. Serial numbers are verified upon return."
    },
    {
        "category": "Shipping & Delivery",
        "title": "Shipping Options and Tracking Orders",
        "content": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Tracking numbers are emailed automatically once an order ships. You can track your package using order IDs formatted like TG-1001 or TG-9999 on our tracking page."
    },
    {
        "category": "Troubleshooting",
        "title": "Wireless Earbuds Pairing and Reset Guide",
        "content": "To pair your TechGear Air Buds (Model TG-EAR-01), hold the case button for 3 seconds until the LED flashes white. If connection fails or sound is distorted, reset them by holding the case button for 10 seconds until the LED turns red, then re-pair."
    },
    {
        "category": "Warranty",
        "title": "Limited Hardware Warranty Information",
        "content": "All TechGear devices include a 1-year limited warranty covering manufacturing defects. Warranty does not cover accidental damage, liquid spills, or unauthorized repairs. To submit a claim, provide your purchase invoice and serial number."
    },
    {
        "category": "Account & Billing",
        "title": "Managing Subscriptions and Canceling Accounts",
        "content": "You can manage or cancel your TechGear Cloud subscription anytime under Account Settings > Billing. Cancellations take effect at the end of the current billing cycle. No partial refunds are issued for mid-month cancellations."
    }
]


def ingest_documents():
    """Generates embeddings and inserts support articles into Supabase."""
    print(f"Starting ingestion for {len(SUPPORT_ARTICLES)} articles...")

    for index, article in enumerate(SUPPORT_ARTICLES, start=1):
        print(f"[{index}/{len(SUPPORT_ARTICLES)}] Embedding: '{article['title']}'...")

        text_to_embed = f"Title: {article['title']}\nContent: {article['content']}"
        
        embedding_vector = embedding_model.embed_query(text_to_embed)

        record = {
            "category": article["category"],
            "title": article["title"],
            "content": article["content"],
            "embedding": embedding_vector
        }

        response = supabase.table("support_articles").insert(record).execute()
        
    print("\n Ingestion complete! All articles successfully stored in Supabase.")


if __name__ == "__main__":
    ingest_documents()