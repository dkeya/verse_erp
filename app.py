import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import hashlib
import os
from dotenv import load_dotenv
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fpdf import FPDF
import plotly.express as px
from streamlit_searchbox import st_searchbox
from typing import Dict, Optional, List, Any, Union, Tuple
from contextlib import contextmanager
import json
import shutil
from pathlib import Path

# Load environment variables
load_dotenv()

# Constants
SYSTEM_DIR = os.getenv("SYSTEM_DIR", os.path.join(os.getcwd(), "system"))
TENANTS_DIR = os.path.join(SYSTEM_DIR, "tenants")
TENANT_CONFIG_FILE = os.path.join(SYSTEM_DIR, "tenants_config.json")
SUBSCRIPTION_PLANS = {
    "starter": {"price_monthly": 49, "price_annual": 499, "features": "Basic features"},
    "growth": {"price_monthly": 99, "price_annual": 999, "features": "Advanced analytics"},
    "enterprise": {"price_monthly": 199, "price_annual": 1999, "features": "Priority support + API"}
}

# Type Aliases
DataFrame = pd.DataFrame
FilePath = Union[str, os.PathLike]

### INITIALIZATION FUNCTIONS ###

def initialize_system() -> None:
    """Initialize the system directory and config file if they don't exist."""
    # Create system directory if it doesn't exist
    os.makedirs(SYSTEM_DIR, exist_ok=True)

    # Create tenants directory if it doesn't exist
    os.makedirs(TENANTS_DIR, exist_ok=True)

    # Initialize tenant config file if it doesn't exist
    if not os.path.exists(TENANT_CONFIG_FILE):
        with open(TENANT_CONFIG_FILE, 'w') as f:
            json.dump({"tenants": {}}, f)

@contextmanager
def file_handler(filename: FilePath, mode: str = 'r'):
    """Context manager for file operations with proper error handling."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Handle case where file doesn't exist for reading
        if 'r' in mode and not os.path.exists(filename):
            if '+' in mode or 'w' in mode or 'a' in mode:
                # Create the file if mode allows writing
                open(filename, 'a').close()
            else:
                raise FileNotFoundError(f"File {filename} does not exist")

        with open(filename, mode) as file:
            yield file
    except Exception as e:
        st.error(f"Error handling file {filename}: {str(e)}")
        raise

def load_tenant_config() -> Dict[str, Any]:
    """Load the tenant configuration."""
    initialize_system()  # Ensure system is initialized first
    try:
        with file_handler(TENANT_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Handle empty/corrupt config file
        return {"tenants": {}}

def save_tenant_config(config: Dict[str, Any]) -> None:
    """Save the tenant configuration."""
    with file_handler(TENANT_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

### TENANT MANAGEMENT FUNCTIONS ###

def get_tenant_dir(tenant_id: str) -> str:
    """Get the directory path for a tenant."""
    return os.path.join(TENANTS_DIR, tenant_id)

def initialize_tenant(tenant_id: str, tenant_name: str, admin_email: str,
                     plan: str = "starter", billing_cycle: str = "monthly") -> None:
    """Initialize a new tenant with required directory structure and files."""
    tenant_dir = get_tenant_dir(tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(tenant_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(tenant_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(tenant_dir, "ecommerce"), exist_ok=True)

    # Initialize tenant config
    config = load_tenant_config()
    config["tenants"][tenant_id] = {
        "name": tenant_name,
        "admin_email": admin_email,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "subscription": {
            "plan": plan,
            "billing_cycle": billing_cycle,
            "status": "trial",
            "trial_end": (datetime.now() + timedelta(days=14)).isoformat(),
            "api_key": None
        }
    }
    save_tenant_config(config)

    # Initialize default admin user
    initialize_default_user(tenant_id)

def get_current_tenant() -> Optional[str]:
    """Get the current tenant ID from session state."""
    return st.session_state.get("current_tenant")

def set_current_tenant(tenant_id: str) -> None:
    """Set the current tenant in session state."""
    st.session_state["current_tenant"] = tenant_id

def get_tenant_file_path(filename: str, tenant_id: Optional[str] = None) -> str:
    """Get the full file path for a tenant-specific file."""
    tenant_id = tenant_id or get_current_tenant()
    if not tenant_id:
        raise ValueError("No tenant ID provided and no current tenant set")
    return os.path.join(get_tenant_dir(tenant_id), filename)

def check_subscription(tenant_id: str) -> bool:
    """Check if tenant has active subscription."""
    config = load_tenant_config()
    tenant = config["tenants"].get(tenant_id, {})

    if not tenant:
        return False

    sub = tenant.get("subscription", {})
    status = sub.get("status", "inactive")

    if status == "active":
        return True
    elif status == "trial":
        trial_end = datetime.fromisoformat(sub["trial_end"])
        return datetime.now() < trial_end
    return False

### UTILITY FUNCTIONS ###

def migrate_single_tenant_data(tenant_id: str) -> None:
    """Migrate data from single-tenant structure to multi-tenant."""
    old_base_dir = os.getenv("BASE_DIR", os.path.join(os.getcwd(), "data"))
    new_tenant_dir = get_tenant_dir(tenant_id)

    if os.path.exists(old_base_dir):
        # Copy all files from old structure to new tenant directory
        for filename in os.listdir(old_base_dir):
            old_path = os.path.join(old_base_dir, filename)
            new_path = os.path.join(new_tenant_dir, filename)

            if os.path.isfile(old_path):
                shutil.copy2(old_path, new_path)
            elif os.path.isdir(old_path) and filename == "images":
                shutil.copytree(old_path, os.path.join(new_tenant_dir, "images"))

        st.success(f"Data migrated successfully to tenant {tenant_id}")

def generate_barcode() -> str:
    """Generate a random barcode."""
    return f"CBW-{random.randint(100000, 999999)}"

def display_dataframe(df: DataFrame, title: str) -> None:
    """Display a DataFrame with a title."""
    st.subheader(title)
    st.dataframe(df)

def save_to_csv(df: DataFrame, filename: str, tenant_id: Optional[str] = None) -> None:
    """Save a DataFrame to a CSV file with tenant awareness."""
    full_path = get_tenant_file_path(filename, tenant_id)
    df.to_csv(full_path, index=False)

def load_from_csv(filename: str, tenant_id: Optional[str] = None) -> DataFrame:
    """Load a DataFrame from a CSV file with tenant awareness."""
    full_path = get_tenant_file_path(filename, tenant_id)
    try:
        return pd.read_csv(full_path)
    except FileNotFoundError:
        return pd.DataFrame()

def scan_barcode(image: Image.Image) -> Optional[str]:
    """Scan a barcode from an image using OpenCV."""
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image_cv)
    return data if data else None

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_default_user(tenant_id: str) -> None:
    """Initialize the default admin user if users.csv doesn't exist."""
    users_file = get_tenant_file_path("users.csv", tenant_id)
    if not os.path.exists(users_file):
        default_user = pd.DataFrame({
            "username": ["admin"],
            "password": [hash_password(os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123"))],
            "role": ["admin"],
            "tenant_id": [tenant_id]
        })
        save_to_csv(default_user, "users.csv", tenant_id)

### AUTHENTICATION FUNCTIONS ###

def authenticate(username: str, password: str, tenant_id: str) -> bool:
    """Authenticate a user within a specific tenant."""
    users_file = get_tenant_file_path("users.csv", tenant_id)
    users_df = pd.read_csv(users_file)
    user_row = users_df[(users_df["username"] == username) & (users_df["tenant_id"] == tenant_id)]

    if not user_row.empty:
        hashed_password = user_row.iloc[0]["password"]
        return hashed_password == hash_password(password)
    return False

def login() -> None:
    """Display the login form with tenant selection."""
    st.sidebar.subheader("Login")

    # Load tenants from config
    config = load_tenant_config()
    tenant_options = {tid: details.get("name", tid) for tid, details in config.get("tenants", {}).items()}

    # No tenants configured yet
    if not tenant_options:
        st.sidebar.warning("No tenants available. Log in as Superadmin to create one.")
        return

    # Tenant selector
    tenant_names = list(tenant_options.values())
    selected_tenant = st.sidebar.selectbox("Select Tenant", tenant_names)

    # Map selected name → tenant_id (safely)
    tenant_id = next((tid for tid, name in tenant_options.items() if name == selected_tenant), None)
    if tenant_id is None:
        st.sidebar.error("Please select a valid tenant from the dropdown.")
        return

    # Credentials
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Login action
    if st.sidebar.button("Login"):
        try:
            if authenticate(username, password, tenant_id):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["login_time"] = datetime.now()
                set_current_tenant(tenant_id)
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid username or password.")
        except FileNotFoundError:
            st.sidebar.error("User database not found for this tenant. Log in as Superadmin and create the tenant first.")

def logout() -> None:
    """Log out the user."""
    if "login_time" in st.session_state:
        logout_time = datetime.now()
        login_time = st.session_state["login_time"]
        hours_worked = (logout_time - login_time).total_seconds() / 3600

        attendance_file = get_tenant_file_path("attendance.csv")
        attendance_df = load_from_csv(attendance_file)

        if attendance_df.empty:
            attendance_df = pd.DataFrame({
                "Attendance ID": [],
                "Employee Name": [],
                "Date": [],
                "Hours Worked": []
            })

        new_attendance = {
            "Attendance ID": f"A{len(attendance_df) + 1:03d}",
            "Employee Name": st.session_state["username"],
            "Date": login_time.strftime("%Y-%m-%d"),
            "Hours Worked": round(hours_worked, 2)
        }
        attendance_df = pd.concat([attendance_df, pd.DataFrame([new_attendance])], ignore_index=True)
        save_to_csv(attendance_df, attendance_file)

    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state.pop("current_tenant", None)
    st.sidebar.success("Logged out successfully!")

### PDF RECEIPT FUNCTION ###

def create_pdf(receipt_data: Dict[str, Any]) -> str:
    """Create a PDF receipt for multiple products using FPDF."""
    pdf = FPDF()
    pdf.add_page()

    # --- Paths (portable) ---
    base_dir = os.path.dirname(__file__)
    assets_dir = os.path.join(base_dir, "assets")
    font_path = os.path.join(assets_dir, "fonts", "DejaVuSans.ttf")

    # Try to find a logo file in assets/
    logo_path = None
    for name in ["logo.png", "logo.jpg", "logo.jpeg", "logo.PNG", "logo.JPG", "logo.JPEG"]:
        candidate = os.path.join(assets_dir, name)
        if os.path.exists(candidate):
            logo_path = candidate
            break

    # --- Font setup (graceful fallback) ---
    try:
        if os.path.exists(font_path):
            pdf.add_font("DejaVuSans", "", font_path, uni=True)
            pdf.add_font("DejaVuSans", "B", font_path, uni=True)
            pdf.add_font("DejaVuSans", "I", font_path, uni=True)
            body_font = ("DejaVuSans", "")
            bold_font = ("DejaVuSans", "B")
            italic_font = ("DejaVuSans", "I")
        else:
            raise FileNotFoundError
    except Exception:
        # Fallback to built-in Helvetica if custom font is missing
        body_font = ("Helvetica", "")
        bold_font = ("Helvetica", "B")
        italic_font = ("Helvetica", "I")

    # --- Layout constants ---
    left_margin = 10
    right_margin = 10
    page_width = 210
    line_width = page_width - left_margin - right_margin

    # --- Logo (optional) ---
    if logo_path:
        pdf.image(logo_path, x=left_margin, y=10, w=20)
    pdf.ln(15)

    def draw_dotted_line(y_position: float, dash_length: int = 2, gap_length: int = 2) -> None:
        x = left_margin
        while x < left_margin + line_width:
            pdf.line(x, y_position, x + dash_length, y_position)
            x += dash_length + gap_length

    # Header
    pdf.set_font(*bold_font, size=12)
    pdf.cell(200, 8, txt="RECEIPT", ln=True, align="C")
    pdf.set_font(*body_font, size=8)

    # Function to draw a dotted line
    def draw_dotted_line(y_position: float, dash_length: int = 2, gap_length: int = 2) -> None:
        x = left_margin
        while x < left_margin + line_width:
            pdf.line(x, y_position, x + dash_length, y_position)
            x += dash_length + gap_length

    # Add receipt content
    pdf.set_font("DejaVuSans", "B", 12)
    pdf.cell(200, 8, txt="RECEIPT", ln=True, align="C")
    pdf.set_font("DejaVuSans", "", 8)

    draw_dotted_line(pdf.get_y() + 3)
    pdf.ln(5)

    # Sale details
    pdf.cell(200, 6, txt=f"Sale ID: {receipt_data['Sale ID']}", ln=True, align="L")
    pdf.cell(200, 6, txt=f"Date: {receipt_data['Date']}", ln=True, align="L")
    pdf.cell(200, 6, txt=f"Customer: {receipt_data['Customer']}", ln=True, align="L")

    draw_dotted_line(pdf.get_y() + 3)
    pdf.ln(5)

    # Product header
    pdf.cell(200, 6, txt="Product       | Quantity | Unit Price  | Total Price", ln=True, align="L")

    draw_dotted_line(pdf.get_y() + 3)
    pdf.ln(5)

    # Add each product to the receipt
    for product in receipt_data["Products"]:
        quantity = receipt_data["Quantities"][product]
        unit_price = receipt_data["Unit Prices"][product]
        total_price = quantity * unit_price
        pdf.cell(200, 6, txt=f"{product}       |   {quantity}      | Kes{unit_price:,.2f}   | Kes{total_price:,.2f}", ln=True, align="L")

    draw_dotted_line(pdf.get_y() + 3)
    pdf.ln(5)

    # Totals
    pdf.cell(200, 6, txt=f"VAT (16%): Kes{receipt_data['VAT']:,.2f}", ln=True, align="L")
    pdf.cell(200, 6, txt=f"Grand Total: Kes{receipt_data['Grand Total']:,.2f}", ln=True, align="L")

    draw_dotted_line(pdf.get_y() + 3)
    pdf.ln(5)

    # Payment method
    pdf.cell(200, 6, txt=f"Payment Method: {receipt_data['Payment Method']}", ln=True, align="L")

    draw_dotted_line(pdf.get_y() + 3)
    pdf.ln(5)

    # Final message
    pdf.set_font("DejaVuSans", "I", 8)
    pdf.cell(200, 6, txt="Thank you for your purchase!", ln=True, align="C")
    pdf.cell(200, 6, txt="Visit us again. 😊", ln=True, align="C")
    pdf.set_font("DejaVuSans", "", 8)

    draw_dotted_line(pdf.get_y() + 3)

    # Save the PDF to a temporary file
    pdf_file = f"receipt_{receipt_data['Sale ID']}.pdf"
    pdf.output(pdf_file)
    return pdf_file

def generate_receipt_text(receipt_data: Dict[str, Any]) -> str:
    """Generate a text-based receipt for multiple products."""
    receipt = f"""
-----------------------------------------------------------
🛒 RECEIPT 🛒
Sale ID: {receipt_data["Sale ID"]}     |   Date: {receipt_data["Date"]}   |   Customer: {receipt_data["Customer"]}
-----------------------------------------------------------
Product       | Quantity | Unit Price  | Total Price
-----------------------------------------------------------
"""
    # Add each product to the receipt
    for product in receipt_data["Products"]:
        quantity = receipt_data["Quantities"][product]
        unit_price = receipt_data["Unit Prices"][product]
        total_price = quantity * unit_price
        receipt += f"{product}       |   {quantity}      | Kes{unit_price:,.2f}   | Kes{total_price:,.2f}\n"

    # Add totals
    receipt += f"""
-----------------------------------------------------------
VAT (16%): Kes{receipt_data["VAT"]:,.2f}          |        Grand Total: Kes{receipt_data["Grand Total"]:,.2f}
-----------------------------------------------------------
Payment Method: {receipt_data["Payment Method"]}
-----------------------------------------------------------
                  Thank you for your purchase!
                    Visit us again. 😊
-----------------------------------------------------------
"""
    return receipt

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send an email using SMTP."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")

    msg = MIMEMultipart()
    msg["From"] = smtp_username
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def subscription_management(tenant_id: str) -> None:
    """Manage tenant subscriptions."""
    st.title("Subscription Management")

    config = load_tenant_config()
    tenant = config["tenants"].get(tenant_id, {})
    subscription = tenant.get("subscription", {})

    # Display current subscription status
    st.subheader("Current Subscription")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Plan", subscription.get("plan", "None"))
        st.metric("Billing Cycle", subscription.get("billing_cycle", "None"))
    with col2:
        st.metric("Status", subscription.get("status", "inactive"))
        if subscription.get("status") == "trial":
            trial_end = datetime.fromisoformat(subscription["trial_end"])
            days_left = (trial_end - datetime.now()).days
            st.metric("Trial Days Remaining", days_left)

    # Subscription upgrade/downgrade
    st.subheader("Change Subscription Plan")
    with st.form("subscription_form"):
        new_plan = st.selectbox("Select Plan", list(SUBSCRIPTION_PLANS.keys()))
        billing_cycle = st.selectbox("Billing Cycle", ["monthly", "annual"])

        if st.form_submit_button("Update Subscription"):
            config["tenants"][tenant_id]["subscription"]["plan"] = new_plan
            config["tenants"][tenant_id]["subscription"]["billing_cycle"] = billing_cycle
            save_tenant_config(config)
            st.success("Subscription plan updated successfully!")

    # Payment processing (simplified)
    st.subheader("Make Payment")
    if subscription["status"] in ["trial", "active"]:
        plan_details = SUBSCRIPTION_PLANS[subscription["plan"]]
        amount = plan_details[f"price_{subscription['billing_cycle']}"]

        st.write(f"Amount Due: ${amount}")
        if st.button("Process Payment"):
            config["tenants"][tenant_id]["subscription"]["status"] = "active"
            config["tenants"][tenant_id]["subscription"]["last_payment"] = datetime.now().isoformat()
            save_tenant_config(config)
            st.success("Payment processed successfully! Subscription activated.")

def tenant_management() -> None:
    """Manage tenant creation and configuration."""
    st.title("Tenant Management")

    if not st.session_state.get("authenticated", False) or st.session_state.get("username") != "superadmin":
        st.error("Only superadmin can access tenant management")
        return

    config = load_tenant_config()

    # Create new tenant
    with st.expander("Create New Tenant"):
        with st.form("tenant_form"):
            tenant_name = st.text_input("Tenant Name", placeholder="Enter tenant name")
            admin_email = st.text_input("Admin Email", placeholder="Enter admin email")
            tenant_id = st.text_input("Tenant ID", placeholder="Enter unique tenant ID")
            plan = st.selectbox("Subscription Plan", list(SUBSCRIPTION_PLANS.keys()))
            billing_cycle = st.selectbox("Billing Cycle", ["monthly", "annual"])

            if st.form_submit_button("Create Tenant"):
                if tenant_id in config["tenants"]:
                    st.error("Tenant ID already exists")
                else:
                    initialize_tenant(tenant_id, tenant_name, admin_email, plan, billing_cycle)
                    st.success(f"Tenant {tenant_name} created successfully!")

    # List existing tenants
    st.subheader("Existing Tenants")
    tenants_df = pd.DataFrame.from_dict(config["tenants"], orient="index")
    st.dataframe(tenants_df)

    # Manage individual tenant subscriptions
    st.subheader("Manage Tenant Subscriptions")
    selected_tenant = st.selectbox("Select Tenant", list(config["tenants"].keys()))
    if selected_tenant:
        subscription_management(selected_tenant)

    # Migration tool for single-tenant data
    with st.expander("Data Migration Tool"):
        st.warning("Use this only once to migrate from single-tenant to multi-tenant")
        tenant_id = st.selectbox("Select Tenant for Migration", list(config["tenants"].keys()))

        if st.button("Migrate Data"):
            migrate_single_tenant_data(tenant_id)

def ecommerce_module(submenu: Optional[str] = None) -> None:
    """E-commerce module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if not check_subscription(tenant_id):
        st.error("Subscription required for e-commerce features")
        return

    st.title("E-commerce Management")

    if submenu == "Product Catalog":
        products_file = get_tenant_file_path("ecommerce/products.csv")
        products_df = load_from_csv(products_file)

        if products_df.empty:
            products_df = pd.DataFrame({
                "Product ID": [],
                "Product Name": [],
                "Description": [],
                "Price": [],
                "Category": [],
                "Stock": [],
                "Image": [],
                "tenant_id": []
            })

        display_dataframe(products_df, "Product Catalog")

        with st.form("product_form"):
            product_name = st.text_input("Product Name")
            description = st.text_area("Description")
            price = st.number_input("Price", min_value=0.01)
            category = st.text_input("Category")
            stock = st.number_input("Stock", min_value=0)
            image = st.file_uploader("Product Image", type=["jpg", "jpeg", "png"])

            if st.form_submit_button("Add Product"):
                new_product = {
                    "Product ID": f"EC{len(products_df) + 1:03d}",
                    "Product Name": product_name,
                    "Description": description,
                    "Price": price,
                    "Category": category,
                    "Stock": stock,
                    "Image": image.name if image else "",
                    "tenant_id": tenant_id
                }
                products_df = pd.concat([products_df, pd.DataFrame([new_product])], ignore_index=True)
                save_to_csv(products_df, "ecommerce/products.csv")

                if image:
                    image_path = get_tenant_file_path(f"ecommerce/images/{image.name}")
                    with open(image_path, "wb") as f:
                        f.write(image.getbuffer())

                st.success("Product added successfully!")

def home() -> None:
    """Home module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    st.image("logo.png", width=150)
    st.title(f"Welcome to the Verse! ({tenant_id})")
    st.write("Your all-in-one solution for managing production, inventory, sales, and more.")

    # Check subscription status
    config = load_tenant_config()
    tenant = config["tenants"].get(tenant_id, {})
    subscription = tenant.get("subscription", {})

    if subscription.get("status") == "trial":
        trial_end = datetime.fromisoformat(subscription["trial_end"])
        days_left = (trial_end - datetime.now()).days
        st.warning(f"You are on a free trial. {days_left} days remaining.")
    elif subscription.get("status") != "active":
        st.error("Your subscription is inactive. Please renew to access all features.")

    sales_df = load_from_csv("sales.csv")
    inventory_df = load_from_csv("inventory.csv")
    production_df = load_from_csv("production.csv")

    total_sales = sales_df["Total Price"].sum() if not sales_df.empty else 0
    inventory_levels = inventory_df["Stock Quantity"].sum() if not inventory_df.empty else 0
    production_batches = len(production_df) if not production_df.empty else 0

    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"Kes {total_sales:,.2f}")
    with col2:
        st.metric("Inventory Levels", f"{inventory_levels:,} Items")
    with col3:
        st.metric("Production Batches", f"{production_batches:,} Batches")

    st.subheader("Quick Links")
    st.write("Navigate to specific modules using the sidebar or the links below:")
    st.write("- [Production Management](#production-management)")
    st.write("- [Inventory Management](#inventory-management)")
    st.write("- [Point of Sale (POS)](#point-of-sale-pos)")
    st.write("- [E-commerce](#e-commerce)")
    st.write("- [Sales & Marketing](#sales-marketing)")
    st.write("- [Analytics & Reporting](#analytics-reporting)")

def production_management(submenu: Optional[str] = None) -> None:
    """Production management module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if submenu == "Production Tracking":
        df = load_from_csv("production.csv")
        inventory_df = load_from_csv("inventory.csv")

        if df.empty:
            df = pd.DataFrame({
                "Batch ID": [], "Product Name": [], "Raw Materials Used": [],
                "Quantity Produced": [], "Unit Price": [], "Production Date": [],
                "Status": [], "tenant_id": []
            })

        display_dataframe(df, "Production Batches")

        with st.form("production_form"):
            product_name = st.text_input("Product Name", placeholder="Enter product name")
            raw_materials = st.text_area("Raw Materials Used", placeholder="List raw materials")
            quantity = st.number_input("Quantity Produced", min_value=1)
            unit_price = st.number_input("Unit Price (KES)", min_value=0.0, value=0.0)
            production_date = st.date_input("Production Date")
            status = st.selectbox("Status", ["Scheduled", "In Progress", "Completed"])
            submit = st.form_submit_button("Add Batch")

            if submit:
                if not product_name or not raw_materials:
                    st.error("Please fill in all fields.")
                else:
                    new_batch = {
                        "Batch ID": f"B{len(df) + 1:03d}",
                        "Product Name": product_name,
                        "Raw Materials Used": raw_materials,
                        "Quantity Produced": quantity,
                        "Unit Price": unit_price,
                        "Production Date": production_date.strftime("%Y-%m-%d"),
                        "Status": status,
                        "tenant_id": tenant_id
                    }
                    df = pd.concat([df, pd.DataFrame([new_batch])], ignore_index=True)
                    save_to_csv(df, "production.csv")

                    # Automatically add to inventory when production is completed
                    if status == "Completed":
                        # Check if product already exists in inventory
                        if product_name in inventory_df["Product Name"].values:
                            # Update existing inventory
                            inventory_df.loc[inventory_df["Product Name"] == product_name, "Stock Quantity"] += quantity
                        else:
                            # Add new product to inventory
                            new_inventory_item = {
                                "Product ID": f"P{len(inventory_df) + 1:03d}",
                                "Product Name": product_name,
                                "Stock Quantity": quantity,
                                "Unit Price": unit_price,
                                "Reorder Level": 10,  # Default value
                                "Last Restocked": production_date.strftime("%Y-%m-%d"),
                                "Expiration Date": (production_date + timedelta(days=365)).strftime("%Y-%m-%d"),
                                "Supplier": "Internal Production",
                                "Barcode": generate_barcode(),
                                "tenant_id": tenant_id
                            }
                            inventory_df = pd.concat([inventory_df, pd.DataFrame([new_inventory_item])], ignore_index=True)

                        save_to_csv(inventory_df, "inventory.csv")
                        st.success(f"✅ {quantity} units of {product_name} added to inventory!")

                    st.success(f"Production batch for {product_name} added successfully!")

    elif submenu == "Workflow Management":
        workflows_file = get_tenant_file_path("workflows.csv")
        workflows_df = load_from_csv(workflows_file)

        if workflows_df.empty:
            workflows_df = pd.DataFrame({
                "Workflow ID": [],
                "Workflow Name": [],
                "Steps": [],
                "Status": [],
                "Start Date": [],
                "End Date": [],
                "tenant_id": []
            })

        display_dataframe(workflows_df, "Workflows")

        st.subheader("Add New Workflow")
        with st.form("workflow_form"):
            workflow_name = st.text_input("Workflow Name", placeholder="Enter workflow name")
            steps = st.text_area("Steps", placeholder="List steps (comma-separated)")
            status = st.selectbox("Status", ["Not Started", "In Progress", "Completed"])
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            submit = st.form_submit_button("Add Workflow")

            if submit:
                if not workflow_name or not steps:
                    st.error("Please fill in all fields.")
                else:
                    new_workflow = {
                        "Workflow ID": f"WF{len(workflows_df) + 1:03d}",
                        "Workflow Name": workflow_name,
                        "Steps": steps,
                        "Status": status,
                        "Start Date": start_date.strftime("%Y-%m-%d"),
                        "End Date": end_date.strftime("%Y-%m-%d"),
                        "tenant_id": tenant_id
                    }
                    workflows_df = pd.concat([workflows_df, pd.DataFrame([new_workflow])], ignore_index=True)
                    save_to_csv(workflows_df, "workflows.csv")
                    st.success(f"Workflow '{workflow_name}' added successfully!")

        st.subheader("Track Workflow Progress")
        if not workflows_df.empty:
            workflow_id = st.selectbox("Select Workflow", workflows_df["Workflow ID"])
            workflow_row = workflows_df[workflows_df["Workflow ID"] == workflow_id].iloc[0]

            st.write(f"**Workflow Name:** {workflow_row['Workflow Name']}")
            st.write(f"**Steps:** {workflow_row['Steps']}")
            st.write(f"**Status:** {workflow_row['Status']}")
            st.write(f"**Start Date:** {workflow_row['Start Date']}")
            st.write(f"**End Date:** {workflow_row['End Date']}")
        else:
            st.warning("No workflows available to track.")

    elif submenu == "Product Formulations":
        formulations_file = get_tenant_file_path("formulations.csv")
        formulations_df = load_from_csv(formulations_file)

        if formulations_df.empty:
            formulations_df = pd.DataFrame({
                "Formulation ID": [],
                "Product Name": [],
                "Ingredients": [],
                "Quantities": [],
                "Unit Price": [],  # Added Unit Price field
                "Version": [],
                "Created By": [],
                "Created Date": [],
                "tenant_id": []
            })

        display_dataframe(formulations_df, "Product Formulations")

        st.subheader("Add New Formulation")
        with st.form("formulation_form"):
            product_name = st.text_input("Product Name", placeholder="Enter product name")
            ingredients = st.text_area("Ingredients", placeholder="List ingredients (comma-separated)")
            quantities = st.text_area("Quantities", placeholder="List quantities (comma-separated)")
            unit_price = st.number_input("Target Unit Price (KES)", min_value=0.0, value=0.0)  # Added Unit Price input
            version = st.text_input("Version", placeholder="Enter version (e.g., v1.0)")
            created_by = st.session_state.get("username", "Admin")
            created_date = datetime.now().strftime("%Y-%m-%d")
            submit = st.form_submit_button("Add Formulation")

            if submit:
                if not product_name or not ingredients or not quantities:
                    st.error("Please fill in all fields.")
                else:
                    new_formulation = {
                        "Formulation ID": f"F{len(formulations_df) + 1:03d}",
                        "Product Name": product_name,
                        "Ingredients": ingredients,
                        "Quantities": quantities,
                        "Unit Price": unit_price,
                        "Version": version,
                        "Created By": created_by,
                        "Created Date": created_date,
                        "tenant_id": tenant_id
                    }
                    formulations_df = pd.concat([formulations_df, pd.DataFrame([new_formulation])], ignore_index=True)
                    save_to_csv(formulations_df, "formulations.csv")
                    st.success(f"Formulation for '{product_name}' added successfully!")

        st.subheader("View Formulation Details")
        if not formulations_df.empty:
            formulation_id = st.selectbox("Select Formulation", formulations_df["Formulation ID"])
            formulation_row = formulations_df[formulations_df["Formulation ID"] == formulation_id].iloc[0]

            st.write(f"**Product Name:** {formulation_row['Product Name']}")
            st.write(f"**Ingredients:** {formulation_row['Ingredients']}")
            st.write(f"**Quantities:** {formulation_row['Quantities']}")
            st.write(f"**Target Unit Price:** KES {formulation_row['Unit Price']:,.2f}")  # Added price display
            st.write(f"**Version:** {formulation_row['Version']}")
            st.write(f"**Created By:** {formulation_row['Created By']}")
            st.write(f"**Created Date:** {formulation_row['Created Date']}")
        else:
            st.warning("No formulations available to view.")

def inventory_management(submenu: Optional[str] = None) -> None:
    """Inventory management module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if submenu == "Stock Levels":
        df = load_from_csv("inventory.csv")
        if df.empty:
            df = pd.DataFrame({
                "Product ID": [], "Product Name": [], "Stock Quantity": [],
                "Unit Price": [], "Reorder Level": [], "Last Restocked": [],
                "Expiration Date": [], "Supplier": [], "Barcode": [], "tenant_id": []
            })

        display_dataframe(df, "Inventory Levels")

        with st.form("inventory_form"):
            product_name = st.text_input("Product Name", placeholder="Enter product name")
            stock_quantity = st.number_input("Stock Quantity", min_value=1)
            unit_price = st.number_input("Unit Price", min_value=0.0, value=0.0)  # ADDED THIS
            reorder_level = st.number_input("Reorder Level", min_value=1)
            last_restocked = st.date_input("Last Restocked Date")
            expiration_date = st.date_input("Expiration Date")
            supplier = st.text_input("Supplier Name", placeholder="Enter supplier name")
            barcode = generate_barcode()
            submit = st.form_submit_button("Add to Inventory")

            if submit:
                if not product_name or not supplier:
                    st.error("Please fill in all fields.")
                elif reorder_level >= stock_quantity:
                    st.error("Reorder Level must be less than Stock Quantity.")
                else:
                    new_item = {
                        "Product ID": f"P{len(df) + 1:03d}",
                        "Product Name": product_name,
                        "Stock Quantity": stock_quantity,
                        "Unit Price": unit_price,  # ADDED THIS
                        "Reorder Level": reorder_level,
                        "Last Restocked": last_restocked.strftime("%Y-%m-%d"),
                        "Expiration Date": expiration_date.strftime("%Y-%m-%d"),
                        "Supplier": supplier,
                        "Barcode": barcode,
                        "tenant_id": tenant_id
                    }
                    df = pd.concat([df, pd.DataFrame([new_item])], ignore_index=True)
                    save_to_csv(df, "inventory.csv")
                    st.success(f"{product_name} added to inventory successfully! Generated Barcode: {barcode}")

    elif submenu == "Reorder Alerts":
        df = load_from_csv("inventory.csv")
        if df.empty:
            st.warning("No inventory data found.")
        else:
            low_stock = df[df["Stock Quantity"] < df["Reorder Level"]]
            if not low_stock.empty:
                st.warning("The following products need reordering:")
                display_dataframe(low_stock, "Low Stock Items")
            else:
                st.success("All stock levels are above reorder levels.")

    elif submenu == "Barcode Management":
        st.write("Scan and manage barcodes.")
        uploaded_file = st.file_uploader("Upload Barcode Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Barcode", use_container_width=True)
            barcode = scan_barcode(image)
            if barcode:
                st.success(f"Scanned Barcode: {barcode}")
            else:
                st.error("No barcode detected.")

def pos_module(submenu: Optional[str] = None) -> None:
    """POS module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if submenu == "Process Sale":
        inventory_df = load_from_csv("inventory.csv")
        sales_df = load_from_csv("sales.csv")

        # Ensure inventory has required columns
        if inventory_df.empty:
            inventory_df = pd.DataFrame(columns=[
                "Product ID", "Product Name", "Stock Quantity",
                "Unit Price", "Barcode", "tenant_id"
            ])
        elif "Unit Price" not in inventory_df.columns:
            inventory_df["Unit Price"] = 0.0  # Add default price if missing

        if sales_df.empty:
            sales_df = pd.DataFrame({
                "Sale ID": [], "Product ID": [], "Product Name": [], "Quantity Sold": [],
                "Unit Price": [], "Total Price": [], "VAT (16%)": [], "Payment Method": [],
                "Sale Date": [], "Customer Name": [], "Status": [], "tenant_id": []
            })

        # Barcode scanning
        uploaded_file = st.file_uploader("Upload Barcode Image", type=["jpg", "jpeg", "png"])
        scanned_product = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Barcode", use_column_width=True)
            barcode = scan_barcode(image)
            if barcode:
                st.success(f"Scanned Barcode: {barcode}")
                scanned_product = inventory_df[inventory_df["Barcode"] == barcode]
                if not scanned_product.empty:
                    st.write(f"Product Found: {scanned_product.iloc[0]['Product Name']}")
                    st.write(f"Price: KES {scanned_product.iloc[0]['Unit Price']:,.2f}")
                else:
                    st.error("No product found with this barcode.")

        # Initialize session state for selected products
        if "selected_products" not in st.session_state:
            st.session_state["selected_products"] = []

        # Process a new sale
        with st.form("pos_form"):
            product_options = ["Select product here"] + inventory_df["Product Name"].unique().tolist()

            selected_products = st.multiselect(
                "Select Products",
                product_options[1:],
                key="selected_products"
            )

            product_quantities = {}
            product_prices = {}
            for product in selected_products:
                product_row = inventory_df[inventory_df["Product Name"] == product].iloc[0]
                max_quantity = float(product_row["Stock Quantity"])

                col1, col2 = st.columns(2)
                with col1:
                    quantity = st.number_input(
                        f"Quantity for {product} (Max: {max_quantity})",
                        min_value=1.0,
                        max_value=max_quantity,
                        value=1.0,
                        step=1.0,
                        key=f"quantity_{product}"
                    )
                with col2:
                    price = st.number_input(
                        f"Price for {product}",
                        min_value=0.0,
                        value=float(product_row["Unit Price"]),
                        key=f"price_{product}"
                    )

                product_quantities[product] = int(quantity)
                product_prices[product] = price

            payment_method = st.selectbox("Payment Method", ["Cash", "Mpesa", "Bank", "Credit Card", "Credit Sales"])
            customer_name = st.text_input("Customer Name", placeholder="Enter customer name")

            submit = st.form_submit_button("Process Sale")
            if submit:
                if not selected_products:
                    st.error("Please select at least one product.")
                elif not customer_name:
                    st.error("Please enter the customer name.")
                else:
                    # Process the sale for each selected product
                    for product in selected_products:
                        product_row = inventory_df[inventory_df["Product Name"] == product].iloc[0]
                        product_id = product_row["Product ID"]
                        unit_price = product_prices[product]
                        quantity_sold = product_quantities[product]
                        total_price = unit_price * quantity_sold
                        vat = total_price * 0.16
                        grand_total = total_price + vat

                        # Update inventory
                        inventory_df.loc[inventory_df["Product Name"] == product, "Stock Quantity"] -= quantity_sold

                        # Update price in inventory if changed
                        if unit_price != product_row["Unit Price"]:
                            inventory_df.loc[inventory_df["Product Name"] == product, "Unit Price"] = unit_price
                            st.info(f"Updated price for {product} to KES {unit_price:,.2f}")

                        # Add sale to sales DataFrame
                        new_sale = {
                            "Sale ID": f"S{len(sales_df) + 1:03d}",
                            "Product ID": product_id,
                            "Product Name": product,
                            "Quantity Sold": quantity_sold,
                            "Unit Price": unit_price,
                            "Total Price": total_price,
                            "VAT (16%)": vat,
                            "Payment Method": payment_method,
                            "Sale Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Customer Name": customer_name,
                            "Status": "Completed",
                            "tenant_id": tenant_id
                        }
                        sales_df = pd.concat([sales_df, pd.DataFrame([new_sale])], ignore_index=True)

                    # Save updated data
                    save_to_csv(inventory_df, "inventory.csv")
                    save_to_csv(sales_df, "sales.csv")
                    st.success("Sale processed successfully!")

                    # Store receipt data in session state
                    st.session_state["receipt"] = {
                        "Sale ID": new_sale["Sale ID"],
                        "Date": new_sale["Sale Date"],
                        "Customer": new_sale["Customer Name"],
                        "Products": selected_products,
                        "Quantities": product_quantities,
                        "Unit Prices": product_prices,
                        "Total Price": sum([product_quantities[product] * product_prices[product] for product in selected_products]),
                        "VAT": sum([product_quantities[product] * product_prices[product] * 0.16 for product in selected_products]),
                        "Grand Total": sum([product_quantities[product] * product_prices[product] * 1.16 for product in selected_products]),
                        "Payment Method": payment_method
                    }

        # Display receipt and actions outside the form
        if "receipt" in st.session_state:
            receipt_data = st.session_state["receipt"]
            receipt = generate_receipt_text(receipt_data)
            st.subheader("Receipt")
            st.code(receipt)

            # Generate and download PDF
            pdf_file = create_pdf(receipt_data)
            with open(pdf_file, "rb") as file:
                st.download_button(
                    label="Download Receipt as PDF",
                    data=file,
                    file_name=f"receipt_{receipt_data['Sale ID']}.pdf",
                    mime="application/pdf"
                )
            # Clean up the temporary PDF file
            os.remove(pdf_file)

            # Print receipt (placeholder)
            if st.button("Print Receipt"):
                st.write("Printing receipt...")

            # Email receipt
            email = st.text_input("Enter customer email to send receipt:")
            if st.button("Email Receipt"):
                if email:
                    if send_email(email, "Your Receipt", receipt):
                        st.success("Receipt sent successfully!")
                    else:
                        st.error("Failed to send receipt.")
                else:
                    st.error("Please enter a valid email address.")

    elif submenu == "View Sales History":
        sales_df = load_from_csv("sales.csv")
        if sales_df.empty:
            st.warning("No sales data available.")
        else:
            display_dataframe(sales_df, "Sales Transactions")
            sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"], format='mixed')
            today_sales = sales_df[sales_df["Sale Date"].dt.date == datetime.today().date()]
            if not today_sales.empty:
                st.write(f"Total Sales Today: KES{today_sales['Total Price'].sum():,.2f}")
                st.write(f"Total VAT Collected: KES{today_sales['VAT (16%)'].sum():,.2f}")
                payment_summary = today_sales.groupby("Payment Method")["Total Price"].sum().reset_index()
                st.write("Payment Method Summary:")
                st.dataframe(payment_summary)
            else:
                st.write("No sales recorded today.")

    elif submenu == "Void Sale":
        sales_df = load_from_csv("sales.csv")
        if sales_df.empty:
            st.warning("No sales data available.")
        else:
            sale_id = st.selectbox("Select Sale ID to Void", sales_df["Sale ID"])
            sale_row = sales_df[sales_df["Sale ID"] == sale_id].iloc[0]

            if sale_row["Status"] == "Voided":
                st.warning("This sale has already been voided.")
            else:
                st.write(f"Sale Details: {sale_row}")
                senior_password = st.text_input("Enter Senior Password", type="password")
                if senior_password == os.getenv("SENIOR_PASSWORD", "senior123"):
                    sales_df.loc[sales_df["Sale ID"] == sale_id, "Status"] = "Voided"
                    inventory_df = load_from_csv("inventory.csv")
                    product_name = sale_row["Product Name"]
                    quantity_sold = sale_row["Quantity Sold"]
                    inventory_df.loc[inventory_df["Product Name"] == product_name, "Stock Quantity"] += quantity_sold
                    save_to_csv(inventory_df, "inventory.csv")
                    save_to_csv(sales_df, "sales.csv")
                    st.success("Sale voided successfully! Stock levels updated.")
                else:
                    st.error("Incorrect password. Only senior staff can void sales.")

def sales_marketing(submenu: Optional[str] = None) -> None:
    """Sales & Marketing module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    sales_df = load_from_csv("sales.csv")
    if sales_df.empty:
        st.warning("No sales data found. Please upload sales data to proceed.")
        return

    try:
        sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"], format='mixed')
    except ValueError as e:
        st.error(f"Error parsing dates: {e}")
        return

    if submenu == "Sales Performance":
        st.subheader("Sales Trends")
        sales_trends = sales_df.set_index("Sale Date")["Total Price"].resample("D").sum()
        st.line_chart(sales_trends)

        st.subheader("Top Products")
        top_products = sales_df["Product Name"].value_counts().reset_index()
        top_products.columns = ["Product Name", "Units Sold"]
        st.bar_chart(top_products.set_index("Product Name"))

    elif submenu == "Customer Insights":
        st.subheader("Customer Insights")
        if "Customer Name" not in sales_df.columns:
            st.error("The 'Customer Name' column is missing in the sales data.")
        else:
            customer_data = sales_df["Customer Name"].value_counts().reset_index()
            customer_data.columns = ["Customer Name", "Number of Purchases"]
            st.dataframe(customer_data)

    elif submenu == "Campaign Management":
        campaigns_file = get_tenant_file_path("campaigns.csv")
        campaigns_df = load_from_csv(campaigns_file)

        if campaigns_df.empty:
            campaigns_df = pd.DataFrame({
                "Campaign ID": [],
                "Campaign Name": [],
                "Start Date": [],
                "End Date": [],
                "Budget": [],
                "Target Audience": [],
                "Reach": [],
                "Engagement": [],
                "ROI": [],
                "tenant_id": []
            })

        display_dataframe(campaigns_df, "Campaigns")

        st.subheader("Create New Campaign")
        with st.form("campaign_form"):
            campaign_name = st.text_input("Campaign Name", placeholder="Enter campaign name")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            budget = st.number_input("Budget", min_value=0)
            target_audience = st.text_input("Target Audience", placeholder="Enter target audience")
            submit = st.form_submit_button("Create Campaign")

            if submit:
                if not campaign_name or not target_audience:
                    st.error("Please fill in all fields.")
                else:
                    new_campaign = {
                        "Campaign ID": f"C{len(campaigns_df) + 1:03d}",
                        "Campaign Name": campaign_name,
                        "Start Date": start_date.strftime("%Y-%m-%d"),
                        "End Date": end_date.strftime("%Y-%m-%d"),
                        "Budget": budget,
                        "Target Audience": target_audience,
                        "Reach": 0,
                        "Engagement": 0,
                        "ROI": 0,
                        "tenant_id": tenant_id
                    }
                    campaigns_df = pd.concat([campaigns_df, pd.DataFrame([new_campaign])], ignore_index=True)
                    save_to_csv(campaigns_df, "campaigns.csv")
                    st.success(f"Campaign '{campaign_name}' created successfully!")

        st.subheader("Track Campaign Performance")
        if not campaigns_df.empty:
            campaign_id = st.selectbox("Select Campaign", campaigns_df["Campaign ID"])
            campaign_row = campaigns_df[campaigns_df["Campaign ID"] == campaign_id].iloc[0]

            st.write(f"**Campaign Name:** {campaign_row['Campaign Name']}")
            st.write(f"**Start Date:** {campaign_row['Start Date']}")
            st.write(f"**End Date:** {campaign_row['End Date']}")
            st.write(f"**Budget:** Kes{campaign_row['Budget']:,.2f}")
            st.write(f"**Target Audience:** {campaign_row['Target Audience']}")
            st.write(f"**Reach:** {campaign_row['Reach']}")
            st.write(f"**Engagement:** {campaign_row['Engagement']}")
            st.write(f"**ROI:** {campaign_row['ROI']}%")
        else:
            st.warning("No campaigns available to track.")

    elif submenu == "Email Marketing":
        email_campaigns_file = get_tenant_file_path("email_campaigns.csv")
        email_campaigns_df = load_from_csv(email_campaigns_file)

        if email_campaigns_df.empty:
            email_campaigns_df = pd.DataFrame({
                "Campaign ID": [],
                "Campaign Name": [],
                "Subject": [],
                "Content": [],
                "Recipients": [],
                "Sent Date": [],
                "Open Rate": [],
                "Click Rate": [],
                "tenant_id": []
            })

        display_dataframe(email_campaigns_df, "Email Campaigns")

        st.subheader("Create New Email Campaign")
        with st.form("email_campaign_form"):
            campaign_name = st.text_input("Campaign Name", placeholder="Enter campaign name")
            subject = st.text_input("Subject", placeholder="Enter email subject")
            content = st.text_area("Content", placeholder="Enter email content")
            recipients = st.text_area("Recipients", placeholder="Enter recipient emails (comma-separated)")
            submit = st.form_submit_button("Create Campaign")

            if submit:
                if not campaign_name or not subject or not content or not recipients:
                    st.error("Please fill in all fields.")
                else:
                    new_email_campaign = {
                        "Campaign ID": f"EC{len(email_campaigns_df) + 1:03d}",
                        "Campaign Name": campaign_name,
                        "Subject": subject,
                        "Content": content,
                        "Recipients": recipients,
                        "Sent Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Open Rate": 0,
                        "Click Rate": 0,
                        "tenant_id": tenant_id
                    }
                    email_campaigns_df = pd.concat([email_campaigns_df, pd.DataFrame([new_email_campaign])], ignore_index=True)
                    save_to_csv(email_campaigns_df, "email_campaigns.csv")
                    st.success(f"Email campaign '{campaign_name}' created successfully!")

        st.subheader("Track Email Campaign Performance")
        if not email_campaigns_df.empty:
            campaign_id = st.selectbox("Select Email Campaign", email_campaigns_df["Campaign ID"])
            campaign_row = email_campaigns_df[email_campaigns_df["Campaign ID"] == campaign_id].iloc[0]

            st.write(f"**Campaign Name:** {campaign_row['Campaign Name']}")
            st.write(f"**Subject:** {campaign_row['Subject']}")
            st.write(f"**Recipients:** {campaign_row['Recipients']}")
            st.write(f"**Sent Date:** {campaign_row['Sent Date']}")
            st.write(f"**Open Rate:** {campaign_row['Open Rate']}%")
            st.write(f"**Click Rate:** {campaign_row['Click Rate']}%")
        else:
            st.warning("No email campaigns available to track.")

    elif submenu == "Loyalty Programs":
        loyalty_programs_file = get_tenant_file_path("loyalty_programs.csv")
        loyalty_programs_df = load_from_csv(loyalty_programs_file)

        if loyalty_programs_df.empty:
            loyalty_programs_df = pd.DataFrame({
                "Program ID": [],
                "Program Name": [],
                "Points per Purchase": [],
                "Discount per Point": [],
                "Active Customers": [],
                "tenant_id": []
            })

        display_dataframe(loyalty_programs_df, "Loyalty Programs")

        st.subheader("Create New Loyalty Program")
        with st.form("loyalty_program_form"):
            program_name = st.text_input("Program Name", placeholder="Enter program name")
            points_per_purchase = st.number_input("Points per Purchase", min_value=1)
            discount_per_point = st.number_input("Discount per Point (Kes)", min_value=0.01)
            submit = st.form_submit_button("Create Program")

            if submit:
                if not program_name:
                    st.error("Please fill in all fields.")
                else:
                    new_loyalty_program = {
                        "Program ID": f"LP{len(loyalty_programs_df) + 1:03d}",
                        "Program Name": program_name,
                        "Points per Purchase": points_per_purchase,
                        "Discount per Point": discount_per_point,
                        "Active Customers": 0,
                        "tenant_id": tenant_id
                    }
                    loyalty_programs_df = pd.concat([loyalty_programs_df, pd.DataFrame([new_loyalty_program])], ignore_index=True)
                    save_to_csv(loyalty_programs_df, "loyalty_programs.csv")
                    st.success(f"Loyalty program '{program_name}' created successfully!")

        st.subheader("Track Loyalty Program Performance")
        if not loyalty_programs_df.empty:
            program_id = st.selectbox("Select Loyalty Program", loyalty_programs_df["Program ID"])
            program_row = loyalty_programs_df[loyalty_programs_df["Program ID"] == program_id].iloc[0]

            st.write(f"**Program Name:** {program_row['Program Name']}")
            st.write(f"**Points per Purchase:** {program_row['Points per Purchase']}")
            st.write(f"**Discount per Point:** Kes{program_row['Discount per Point']:,.2f}")
            st.write(f"**Active Customers:** {program_row['Active Customers']}")
        else:
            st.warning("No loyalty programs available to track.")

    elif submenu == "Discount Management":
        discounts_file = get_tenant_file_path("discounts.csv")
        discounts_df = load_from_csv(discounts_file)

        if discounts_df.empty:
            discounts_df = pd.DataFrame({
                "Discount ID": [],
                "Discount Name": [],
                "Discount Type": [],
                "Value": [],
                "Start Date": [],
                "End Date": [],
                "Applicable Products": [],
                "tenant_id": []
            })

        display_dataframe(discounts_df, "Discounts")

        st.subheader("Create New Discount")
        with st.form("discount_form"):
            discount_name = st.text_input("Discount Name", placeholder="Enter discount name")
            discount_type = st.selectbox("Discount Type", ["Percentage", "Fixed Amount"])
            value = st.number_input("Value", min_value=0.01)
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            applicable_products = st.text_area("Applicable Products", placeholder="Enter product names (comma-separated)")
            submit = st.form_submit_button("Create Discount")

            if submit:
                if not discount_name or not applicable_products:
                    st.error("Please fill in all fields.")
                else:
                    new_discount = {
                        "Discount ID": f"D{len(discounts_df) + 1:03d}",
                        "Discount Name": discount_name,
                        "Discount Type": discount_type,
                        "Value": value,
                        "Start Date": start_date.strftime("%Y-%m-%d"),
                        "End Date": end_date.strftime("%Y-%m-%d"),
                        "Applicable Products": applicable_products,
                        "tenant_id": tenant_id
                    }
                    discounts_df = pd.concat([discounts_df, pd.DataFrame([new_discount])], ignore_index=True)
                    save_to_csv(discounts_df, "discounts.csv")
                    st.success(f"Discount '{discount_name}' created successfully!")

        st.subheader("Track Discount Performance")
        if not discounts_df.empty:
            discount_id = st.selectbox("Select Discount", discounts_df["Discount ID"])
            discount_row = discounts_df[discounts_df["Discount ID"] == discount_id].iloc[0]

            st.write(f"**Discount Name:** {discount_row['Discount Name']}")
            st.write(f"**Discount Type:** {discount_row['Discount Type']}")
            st.write(f"**Value:** {discount_row['Value']}")
            st.write(f"**Start Date:** {discount_row['Start Date']}")
            st.write(f"**End Date:** {discount_row['End Date']}")
            st.write(f"**Applicable Products:** {discount_row['Applicable Products']}")
        else:
            st.warning("No discounts available to track.")

    elif submenu == "CRM":
        crm_file = get_tenant_file_path("crm.csv")
        crm_df = load_from_csv(crm_file)

        if crm_df.empty:
            crm_df = pd.DataFrame({
                "Customer ID": [],
                "Customer Name": [],
                "Email": [],
                "Phone": [],
                "Purchase History": [],
                "Preferences": [],
                "Last Interaction": [],
                "tenant_id": []
            })

        # Combine Customer ID and Phone for display
        crm_df["Customer Display"] = crm_df["Customer ID"].astype(str) + " - " + crm_df["Phone"].astype(str)

        display_dataframe(crm_df, "Customer Interactions")

        st.subheader("Add New Customer Interaction")
        with st.form("crm_form"):
            customer_name = st.text_input("Customer Name", placeholder="Enter customer name")
            email = st.text_input("Email", placeholder="Enter customer email")
            phone = st.text_input("Phone", placeholder="Enter customer phone")
            purchase_history = st.text_area("Purchase History", placeholder="Enter purchase history")
            preferences = st.text_area("Preferences", placeholder="Enter customer preferences")
            last_interaction = st.date_input("Last Interaction Date")
            submit = st.form_submit_button("Add Interaction")

            if submit:
                if not customer_name or not email or not phone:
                    st.error("Please fill in all fields.")
                else:
                    new_interaction = {
                        "Customer ID": f"C{len(crm_df) + 1:03d}",
                        "Customer Name": customer_name,
                        "Email": email,
                        "Phone": phone,
                        "Purchase History": purchase_history,
                        "Preferences": preferences,
                        "Last Interaction": last_interaction.strftime("%Y-%m-%d"),
                        "tenant_id": tenant_id
                    }
                    crm_df = pd.concat([crm_df, pd.DataFrame([new_interaction])], ignore_index=True)
                    save_to_csv(crm_df, "crm.csv")
                    st.success(f"Interaction for {customer_name} added successfully!")

        st.subheader("Track Customer Interactions")
        if not crm_df.empty:
            def search_customers(search_term: str) -> List[str]:
                return crm_df[crm_df["Customer Display"].str.contains(search_term, case=False)]["Customer Display"].tolist()

            customer_display = st_searchbox(
                search_customers,
                placeholder="Search by Customer ID or Phone",
                label="Select Customer"
            )

            if customer_display:
                customer_id = customer_display.split(" - ")[0]
                customer_row = crm_df[crm_df["Customer ID"] == customer_id].iloc[0]

                st.write(f"**Customer Name:** {customer_row['Customer Name']}")
                st.write(f"**Email:** {customer_row['Email']}")
                st.write(f"**Phone:** {customer_row['Phone']}")
                st.write(f"**Purchase History:** {customer_row['Purchase History']}")
                st.write(f"**Preferences:** {customer_row['Preferences']}")
                st.write(f"**Last Interaction:** {customer_row['Last Interaction']}")
        else:
            st.warning("No customer interactions available to track.")

def personnel_management(submenu: Optional[str] = None) -> None:
    """Personnel management module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if submenu == "Employee Records":
        df = load_from_csv("employees.csv")
        if df.empty:
            df = pd.DataFrame({
                "Employee ID": [], "Employee Name": [], "Role": [],
                "Salary": [], "Join Date": [], "Attendance": [],
                "tenant_id": []
            })

        display_dataframe(df, "Employee Records")

        with st.form("employee_form"):
            employee_name = st.text_input("Employee Name", placeholder="Enter employee name")
            role = st.selectbox("Role", ["Production", "Sales", "HR", "Finance"])
            salary = st.number_input("Salary", min_value=0)
            join_date = st.date_input("Join Date")
            attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
            submit = st.form_submit_button("Add Employee")

            if submit:
                if not employee_name:
                    st.error("Please fill in all fields.")
                else:
                    new_employee = {
                        "Employee ID": f"E{len(df) + 1:03d}",
                        "Employee Name": employee_name,
                        "Role": role,
                        "Salary": salary,
                        "Join Date": join_date.strftime("%Y-%m-%d"),
                        "Attendance": attendance,
                        "tenant_id": tenant_id
                    }
                    df = pd.concat([df, pd.DataFrame([new_employee])], ignore_index=True)
                    save_to_csv(df, "employees.csv")
                    st.success(f"Employee {employee_name} added successfully!")

    elif submenu == "Payroll Processing":
        employees_file = get_tenant_file_path("employees.csv")
        payroll_file = get_tenant_file_path("payroll.csv")
        employees_df = load_from_csv(employees_file)
        payroll_df = load_from_csv(payroll_file)

        if payroll_df.empty:
            payroll_df = pd.DataFrame({
                "Payroll ID": [],
                "Employee ID": [],
                "Employee Name": [],
                "Salary": [],
                "Deductions": [],
                "Net Pay": [],
                "Payment Date": [],
                "tenant_id": []
            })

        display_dataframe(payroll_df, "Payroll Records")

        st.subheader("Process Payroll")
        with st.form("payroll_form"):
            employee_name = st.selectbox("Select Employee", employees_df["Employee Name"])
            employee_row = employees_df[employees_df["Employee Name"] == employee_name].iloc[0]
            salary = employee_row["Salary"]
            deductions = st.number_input("Deductions", min_value=0, value=0)
            net_pay = salary - deductions
            payment_date = st.date_input("Payment Date")
            submit = st.form_submit_button("Process Payroll")

            if submit:
                new_payroll = {
                    "Payroll ID": f"PY{len(payroll_df) + 1:03d}",
                    "Employee ID": employee_row["Employee ID"],
                    "Employee Name": employee_name,
                    "Salary": salary,
                    "Deductions": deductions,
                    "Net Pay": net_pay,
                    "Payment Date": payment_date.strftime("%Y-%m-%d"),
                    "tenant_id": tenant_id
                }
                payroll_df = pd.concat([payroll_df, pd.DataFrame([new_payroll])], ignore_index=True)
                save_to_csv(payroll_df, "payroll.csv")
                st.success(f"Payroll for {employee_name} processed successfully!")

    elif submenu == "Attendance Tracking":
        employees_file = get_tenant_file_path("employees.csv")
        employees_df = load_from_csv(employees_file)

        if employees_df.empty:
            st.warning("No employee data found. Please add employees first.")
            return

        attendance_file = get_tenant_file_path("attendance.csv")
        attendance_df = load_from_csv(attendance_file)

        if attendance_df.empty:
            attendance_df = pd.DataFrame({
                "Attendance ID": [],
                "Employee Name": [],
                "Date": [],
                "Hours Worked": [],
                "tenant_id": []
            })

        display_dataframe(attendance_df, "Attendance Records")

        st.subheader("Log Attendance")
        with st.form("attendance_form"):
            employee_name = st.selectbox("Select Employee", employees_df["Employee Name"])
            date = st.date_input("Date")
            status = st.selectbox("Status", ["Present", "Absent", "Late"])
            hours_worked = st.number_input("Hours Worked", min_value=0, max_value=24, value=8)
            submit = st.form_submit_button("Log Attendance")

            if submit:
                new_attendance = {
                    "Attendance ID": f"A{len(attendance_df) + 1:03d}",
                    "Employee Name": employee_name,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Hours Worked": hours_worked,
                    "tenant_id": tenant_id
                }
                attendance_df = pd.concat([attendance_df, pd.DataFrame([new_attendance])], ignore_index=True)
                save_to_csv(attendance_df, "attendance.csv")
                st.success(f"Attendance for {employee_name} logged successfully!")

    elif submenu == "Performance Reviews":
        reviews_file = get_tenant_file_path("performance_reviews.csv")
        reviews_df = load_from_csv(reviews_file)

        if reviews_df.empty:
            reviews_df = pd.DataFrame({
                "Review ID": [],
                "Employee Name": [],
                "Review Date": [],
                "Reviewer": [],
                "Rating": [],
                "Comments": [],
                "tenant_id": []
            })

        display_dataframe(reviews_df, "Performance Reviews")

        st.subheader("Add New Performance Review")
        with st.form("review_form"):
            employee_name = st.text_input("Employee Name", placeholder="Enter employee name")
            review_date = st.date_input("Review Date")
            reviewer = st.text_input("Reviewer", placeholder="Enter reviewer name")
            rating = st.number_input("Rating (1-5)", min_value=1, max_value=5, value=3)
            comments = st.text_area("Comments", placeholder="Enter comments")
            submit = st.form_submit_button("Add Review")

            if submit:
                if not employee_name or not reviewer:
                    st.error("Please fill in all fields.")
                else:
                    new_review = {
                        "Review ID": f"RV{len(reviews_df) + 1:03d}",
                        "Employee Name": employee_name,
                        "Review Date": review_date.strftime("%Y-%m-%d"),
                        "Reviewer": reviewer,
                        "Rating": rating,
                        "Comments": comments,
                        "tenant_id": tenant_id
                    }
                    reviews_df = pd.concat([reviews_df, pd.DataFrame([new_review])], ignore_index=True)
                    save_to_csv(reviews_df, "performance_reviews.csv")
                    st.success(f"Performance review for {employee_name} added successfully!")

    elif submenu == "Leave Management":
        leave_file = get_tenant_file_path("leave_requests.csv")
        leave_df = load_from_csv(leave_file)

        if leave_df.empty:
            leave_df = pd.DataFrame({
                "Leave ID": [],
                "Employee Name": [],
                "Leave Type": [],
                "Start Date": [],
                "End Date": [],
                "Status": [],
                "tenant_id": []
            })

        display_dataframe(leave_df, "Leave Requests")

        st.subheader("Apply for Leave")
        with st.form("leave_form"):
            employee_name = st.text_input("Employee Name", placeholder="Enter employee name")
            leave_type = st.selectbox("Leave Type", ["Sick Leave", "Vacation", "Maternity Leave", "Other"])
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            submit = st.form_submit_button("Submit Leave Request")

            if submit:
                if not employee_name:
                    st.error("Please fill in all fields.")
                else:
                    new_leave = {
                        "Leave ID": f"LV{len(leave_df) + 1:03d}",
                        "Employee Name": employee_name,
                        "Leave Type": leave_type,
                        "Start Date": start_date.strftime("%Y-%m-%d"),
                        "End Date": end_date.strftime("%Y-%m-%d"),
                        "Status": "Pending",
                        "tenant_id": tenant_id
                    }
                    leave_df = pd.concat([leave_df, pd.DataFrame([new_leave])], ignore_index=True)
                    save_to_csv(leave_df, "leave_requests.csv")
                    st.success(f"Leave request for {employee_name} submitted successfully!")

        st.subheader("Manage Leave Requests")
        if not leave_df.empty:
            leave_id = st.selectbox("Select Leave Request", leave_df["Leave ID"])
            leave_row = leave_df[leave_df["Leave ID"] == leave_id].iloc[0]

            st.write(f"**Employee Name:** {leave_row['Employee Name']}")
            st.write(f"**Leave Type:** {leave_row['Leave Type']}")
            st.write(f"**Start Date:** {leave_row['Start Date']}")
            st.write(f"**End Date:** {leave_row['End Date']}")
            st.write(f"**Status:** {leave_row['Status']}")

            if st.button("Approve Leave"):
                leave_df.loc[leave_df["Leave ID"] == leave_id, "Status"] = "Approved"
                save_to_csv(leave_df, "leave_requests.csv")
                st.success(f"Leave request {leave_id} approved!")

            if st.button("Reject Leave"):
                leave_df.loc[leave_df["Leave ID"] == leave_id, "Status"] = "Rejected"
                save_to_csv(leave_df, "leave_requests.csv")
                st.success(f"Leave request {leave_id} rejected!")

    elif submenu == "Shift Scheduling":
        shifts_file = get_tenant_file_path("shifts.csv")
        shifts_df = load_from_csv(shifts_file)

        if shifts_df.empty:
            shifts_df = pd.DataFrame({
                "Shift ID": [],
                "Employee Name": [],
                "Shift Date": [],
                "Shift Type": [],
                "Start Time": [],
                "End Time": [],
                "tenant_id": []
            })

        display_dataframe(shifts_df, "Shifts")

        st.subheader("Create New Shift")
        with st.form("shift_form"):
            employee_name = st.text_input("Employee Name", placeholder="Enter employee name")
            shift_date = st.date_input("Shift Date")
            shift_type = st.selectbox("Shift Type", ["Morning", "Afternoon", "Night"])
            start_time = st.time_input("Start Time")
            end_time = st.time_input("End Time")
            submit = st.form_submit_button("Add Shift")

            if submit:
                if not employee_name:
                    st.error("Please fill in all fields.")
                else:
                    new_shift = {
                        "Shift ID": f"SH{len(shifts_df) + 1:03d}",
                        "Employee Name": employee_name,
                        "Shift Date": shift_date.strftime("%Y-%m-%d"),
                        "Shift Type": shift_type,
                        "Start Time": start_time.strftime("%H:%M"),
                        "End Time": end_time.strftime("%H:%M"),
                        "tenant_id": tenant_id
                    }
                    shifts_df = pd.concat([shifts_df, pd.DataFrame([new_shift])], ignore_index=True)
                    save_to_csv(shifts_df, "shifts.csv")
                    st.success(f"Shift for {employee_name} added successfully!")

def financial_management(submenu: Optional[str] = None) -> None:
    """Financial management module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if submenu == "Revenue Tracking":
        df = load_from_csv("financial.csv")
        if df.empty:
            df = pd.DataFrame({
                "Transaction ID": [], "Description": [], "Amount": [],
                "Type": [], "Date": [], "tenant_id": []
            })
        revenue_df = df[df["Type"] == "Revenue"]
        display_dataframe(revenue_df, "Revenue Transactions")

    elif submenu == "Expense Tracking":
        df = load_from_csv("financial.csv")
        if df.empty:
            df = pd.DataFrame({
                "Transaction ID": [], "Description": [], "Amount": [],
                "Type": [], "Date": [], "tenant_id": []
            })
        expense_df = df[df["Type"] == "Expense"]
        display_dataframe(expense_df, "Expense Transactions")

    elif submenu == "Tax Management":
        sales_df = load_from_csv("sales.csv")
        if sales_df.empty:
            st.warning("No sales data found. Please upload sales data to proceed.")
            return

        try:
            sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"], format='mixed')
        except Exception as e:
            st.error(f"Error parsing 'Sale Date': {e}")
            return

        if sales_df["Total Price"].isnull().any() or sales_df["Sale Date"].isnull().any():
            st.warning("Missing values found in 'Total Price' or 'Sale Date'. Dropping rows with missing values.")
            sales_df = sales_df.dropna(subset=["Total Price", "Sale Date"])

        if sales_df.empty:
            st.error("No valid data available after handling missing values.")
            return

        sales_df["Tax"] = sales_df["Total Price"] * 0.16
        cumulative_taxes = sales_df["Tax"].sum()
        processed_taxes = sales_df[sales_df["Status"] == "Processed"]["Tax"].sum()
        outstanding_taxes = cumulative_taxes - processed_taxes

        st.subheader("Tax Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cumulative Taxes", f"Kes {cumulative_taxes:,.2f}")
        with col2:
            st.metric("Processed Taxes", f"Kes {processed_taxes:,.2f}")
        with col3:
            st.metric("Outstanding Taxes", f"Kes {outstanding_taxes:,.2f}")

        st.subheader("Process Taxes")
        if outstanding_taxes > 0:
            with st.form("tax_form"):
                default_tax_amount = max(0.01, outstanding_taxes)
                tax_amount = st.number_input(
                    "Tax Amount to Process",
                    min_value=0.01,
                    max_value=outstanding_taxes,
                    value=default_tax_amount
                )
                submit = st.form_submit_button("Process Tax")

                if submit:
                    if tax_amount > outstanding_taxes:
                        st.error("Tax amount to process cannot exceed outstanding taxes.")
                    else:
                        sales_df.loc[sales_df["Status"] != "Processed", "Status"] = "Processed"
                        save_to_csv(sales_df, "sales.csv")
                        st.success(f"Taxes of Kes {tax_amount:,.2f} processed successfully!")
        else:
            st.info("No outstanding taxes to process.")

    elif submenu == "Financial Reports":
        financial_df = load_from_csv("financial.csv")
        if financial_df.empty:
            st.warning("No financial data found. Please upload financial data to proceed.")
            return

        try:
            financial_df["Date"] = pd.to_datetime(financial_df["Date"], format='mixed')
        except Exception as e:
            st.error(f"Error parsing 'Date': {e}")
            return

        if financial_df["Amount"].isnull().any() or financial_df["Date"].isnull().any():
            st.warning("Missing values found in 'Amount' or 'Date'. Dropping rows with missing values.")
            financial_df = financial_df.dropna(subset=["Amount", "Date"])

        if financial_df.empty:
            st.error("No valid data available after handling missing values.")
            return

        st.subheader("Balance Sheet")
        balance_sheet = financial_df.groupby("Type")["Amount"].sum().reset_index()
        st.dataframe(balance_sheet)

        st.subheader("Income Statement")
        income_statement = financial_df.groupby("Type")["Amount"].sum().reset_index()
        st.dataframe(income_statement)

        st.subheader("Cash Flow Statement")
        cash_flow_statement = financial_df.groupby("Type")["Amount"].sum().reset_index()
        st.dataframe(cash_flow_statement)

        st.subheader("Financial Overview")
        fig = px.pie(
            financial_df,
            names="Type",
            values="Amount",
            title="Revenue vs Expenses Breakdown"
        )
        st.plotly_chart(fig)

def analytics_reporting(submenu: Optional[str] = None) -> None:
    """Analytics & Reporting module with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    if submenu == "Sales Analytics":
        sales_revenue_tracking()

    elif submenu == "Customer Segmentation":
        sales_df = load_from_csv("sales.csv")
        if sales_df.empty:
            st.warning("No sales data found. Please upload sales data to proceed.")
            return

        if "Customer Name" not in sales_df.columns or "Total Price" not in sales_df.columns:
            st.error("Required columns ('Customer Name' or 'Total Price') are missing in the sales data.")
            return

        customer_spending = sales_df.groupby("Customer Name")["Total Price"].sum().reset_index()
        customer_spending.rename(columns={"Total Price": "Total Spending"}, inplace=True)

        bins = [0, 1000, 5000, float('inf')]
        labels = ["Low Spenders", "Medium Spenders", "High Spenders"]
        customer_spending["Segment"] = pd.cut(customer_spending["Total Spending"], bins=bins, labels=labels)

        st.subheader("Customer Segments")
        st.dataframe(customer_spending)

        segment_counts = customer_spending["Segment"].value_counts()
        fig = px.bar(
            segment_counts,
            x=segment_counts.index,
            y=segment_counts.values,
            labels={"x": "Segment", "y": "Number of Customers"},
            title="Customer Segment Distribution",
            color=segment_counts.index,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig)

    elif submenu == "Inventory Analytics":
        inventory_df = load_from_csv("inventory.csv")
        if inventory_df.empty:
            st.warning("No inventory data found. Please upload inventory data to proceed.")
            return

        st.subheader("Stock Levels")
        st.bar_chart(inventory_df.set_index("Product Name")["Stock Quantity"])

    elif submenu == "Financial Analytics":
        financial_df = load_from_csv("financial.csv")
        if financial_df.empty:
            st.warning("No financial data found. Please upload financial data to proceed.")
            return

        revenue = financial_df[financial_df["Type"] == "Revenue"]["Amount"].sum()
        expenses = financial_df[financial_df["Type"] == "Expense"]["Amount"].sum()
        st.write(f"**Total Revenue:** Kes{revenue:,.2f}")
        st.write(f"**Total Expenses:** Kes{expenses:,.2f}")
        st.write(f"**Net Profit:** Kes{revenue - expenses:,.2f}")

        st.subheader("Revenue vs Expenses Breakdown")
        labels = ["Revenue", "Expenses"]
        sizes = [revenue, expenses]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    elif submenu == "Sales Forecasting":
        st.title("Sales Forecasting")

        sales_df = load_from_csv("sales.csv")
        if sales_df.empty:
            st.warning("No sales data found. Please upload sales data to proceed.")
            return

        try:
            sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"], format='mixed')
        except Exception as e:
            st.error(f"Error parsing 'Sale Date': {e}")
            return

        if sales_df["Total Price"].isnull().any() or sales_df["Sale Date"].isnull().any():
            st.warning("Missing values found in 'Total Price' or 'Sale Date'. Dropping rows with missing values.")
            sales_df = sales_df.dropna(subset=["Total Price", "Sale Date"])

        if sales_df.empty:
            st.error("No valid data available after handling missing values.")
            return

        sales_df = sales_df.groupby("Sale Date")["Total Price"].sum().reset_index()
        sales_df.set_index("Sale Date", inplace=True)

        sales_trends = sales_df["Total Price"].resample("D").sum().fillna(0)

        model = ExponentialSmoothing(sales_trends, trend="add", seasonal="add", seasonal_periods=7)
        fit = model.fit()

        forecast_periods = st.number_input("Enter number of days to forecast:", min_value=1, value=30)
        forecast = fit.forecast(steps=forecast_periods)

        st.write(f"Forecasted sales for the next {forecast_periods} days:")
        st.line_chart(forecast)

    elif submenu == "Customer Lifetime Value (CLV)":
        sales_df = load_from_csv("sales.csv")
        if sales_df.empty:
            st.warning("No sales data found. Please upload sales data to proceed.")
            return

        if "Customer Name" not in sales_df.columns or "Total Price" not in sales_df.columns:
            st.error("Required columns ('Customer Name' or 'Total Price') are missing in the sales data.")
            return

        try:
            sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"], format='mixed')
        except Exception as e:
            st.error(f"Error parsing 'Sale Date': {e}")
            return

        avg_purchase_value = sales_df["Total Price"].mean()
        purchase_frequency = sales_df.groupby("Customer Name")["Sale ID"].nunique().mean()
        customer_lifespan = (sales_df["Sale Date"].max() - sales_df["Sale Date"].min()).days / 365
        clv = (avg_purchase_value * purchase_frequency) * customer_lifespan

        st.subheader("Customer Lifetime Value (CLV) Insights")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Purchase Value", f"Kes {avg_purchase_value:,.2f}")
        with col2:
            st.metric("Purchase Frequency", f"{purchase_frequency:.2f} purchases/customer")
        with col3:
            st.metric("Customer Lifespan", f"{customer_lifespan:.2f} years")
        with col4:
            st.metric("Customer Lifetime Value (CLV)", f"Kes {clv:,.2f}")

        st.subheader("CLV Components Breakdown")
        clv_components = pd.DataFrame({
            "Component": ["Average Purchase Value", "Purchase Frequency", "Customer Lifespan"],
            "Value": [avg_purchase_value, purchase_frequency, customer_lifespan]
        })

        fig = px.bar(
            clv_components,
            x="Component",
            y="Value",
            title="Components of Customer Lifetime Value (CLV)",
            labels={"Value": "Value", "Component": "Component"},
            color="Component",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig)

        st.subheader("What Does This Mean?")
        st.write(f"""
        - **Average Purchase Value**: The average amount a customer spends per purchase is **Kes {avg_purchase_value:,.2f}**.
        - **Purchase Frequency**: On average, each customer makes **{purchase_frequency:.2f} purchases** over their lifetime.
        - **Customer Lifespan**: The average customer stays with us for **{customer_lifespan:.2f} years**.
        - **Customer Lifetime Value (CLV)**: Based on these metrics, the average customer is worth **Kes {clv:,.2f}** over their lifetime.
        """)

        st.subheader("How Can We Improve CLV?")
        st.write("""
        - **Increase Purchase Frequency**: Encourage repeat purchases through loyalty programs, discounts, or personalized offers.
        - **Increase Average Purchase Value**: Upsell or cross-sell higher-value products to customers.
        - **Extend Customer Lifespan**: Improve customer retention through excellent service, engagement, and satisfaction.
        """)

def sales_revenue_tracking() -> None:
    """Track sales revenue with tenant awareness."""
    tenant_id = get_current_tenant()
    if not tenant_id:
        st.error("No tenant selected")
        return

    sales_file = get_tenant_file_path("sales.csv")
    sales_df = load_from_csv(sales_file)

    if sales_df.empty:
        st.warning("No sales data found. Please upload sales data to proceed.")
        return

    try:
        sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"], format='mixed')
    except Exception as e:
        st.error(f"Error parsing 'Sale Date': {e}")
        return

    if sales_df["Total Price"].isnull().any() or sales_df["Sale Date"].isnull().any():
        st.warning("Missing values found in 'Total Price' or 'Sale Date'. Dropping rows with missing values.")
        sales_df = sales_df.dropna(subset=["Total Price", "Sale Date"])

    if sales_df.empty:
        st.error("No valid data available after handling missing values.")
        return

    sales_df = sales_df.groupby("Sale Date")["Total Price"].sum().reset_index()
    sales_df.set_index("Sale Date", inplace=True)

    daily_revenue = sales_df["Total Price"].resample("D").sum()
    weekly_revenue = sales_df["Total Price"].resample("W").sum()
    monthly_revenue = sales_df["Total Price"].resample("M").sum()
    current_year = datetime.now().year
    ytd_revenue = sales_df[sales_df.index.year == current_year]["Total Price"].resample("M").sum()
    cumulative_revenue = sales_df["Total Price"].cumsum()

    st.subheader("Sales Revenue Summary Table")
    summary_data = {
        "Time Period": ["Total Revenue (Kes)"],
        "Daily": [daily_revenue.sum()],
        "Weekly": [weekly_revenue.sum()],
        "Monthly": [monthly_revenue.sum()],
        "Year-to-Date": [ytd_revenue.sum()],
        "Inception-to-Date": [cumulative_revenue.iloc[-1]]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    revenue_data = pd.DataFrame({
        "Daily Revenue": daily_revenue,
        "Weekly Revenue": weekly_revenue,
        "Monthly Revenue": monthly_revenue,
        "Year-to-Date Revenue": ytd_revenue,
        "Cumulative Revenue": cumulative_revenue
    }).fillna(0)

    fig = px.line(
        revenue_data,
        x=revenue_data.index,
        y=["Daily Revenue", "Weekly Revenue", "Monthly Revenue", "Year-to-Date Revenue", "Cumulative Revenue"],
        title="Sales Revenue Trends Over Time",
        labels={"value": "Revenue (Kes)", "index": "Date"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        legend_title="Time Period",
        xaxis_title="Date",
        yaxis_title="Revenue (Kes)",
        hovermode="x unified"
    )
    st.plotly_chart(fig)

def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="🌐the Verse", layout="wide")

    # Initialize system directories
    os.makedirs(SYSTEM_DIR, exist_ok=True)
    os.makedirs(TENANTS_DIR, exist_ok=True)

    # Check for superadmin (system-wide admin)
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Superadmin login (bypasses tenant login)
    if not st.session_state["authenticated"] and st.sidebar.checkbox("Superadmin Login"):
        superadmin_pw = st.sidebar.text_input("Superadmin Password", type="password")
        if st.sidebar.button("Login as Superadmin"):
            if hash_password(superadmin_pw) == hash_password(os.getenv("SUPERADMIN_PASSWORD", "superadmin123")):
                st.session_state["authenticated"] = True
                st.session_state["username"] = "superadmin"
                st.session_state["login_time"] = datetime.now()
                st.sidebar.success("Logged in as superadmin!")
            else:
                st.sidebar.error("Invalid superadmin password")

    # Regular tenant login
    if not st.session_state.get("authenticated", False):
        login()
        return

    # Superadmin gets tenant management instead of regular modules
    if st.session_state.get("username") == "superadmin":
        tenant_management()
        return

    # Regular user interface
    st.sidebar.title("🌐the Verse")
    menu = {
        "Home": home,
        "Production Management": production_management,
        "Inventory Management": inventory_management,
        "Point of Sale (POS)": pos_module,
        "E-commerce": ecommerce_module,
        "Sales & Marketing": sales_marketing,
        "Personnel Management": personnel_management,
        "Financial Management": financial_management,
        "Analytics & Reporting": analytics_reporting,
    }

    choice = st.sidebar.radio("Select Module", list(menu.keys()), key="main_menu")

    # Add subscription management link for tenant admins
    if st.session_state.get("username") == "admin":
        if st.sidebar.button("Subscription Management"):
            subscription_management(get_current_tenant())

    if st.sidebar.button("Logout", key="logout_button"):
        logout()

    st.sidebar.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: small;
            font-style: italic;
            padding: 10px;
            background-color: transparent;
        }
        </style>
        <div class="footer">powered by Virtual Analytics</div>
        """,
        unsafe_allow_html=True
    )

    if choice != "Home":
        st.title(choice)

    if choice in ["Production Management", "Inventory Management", "Point of Sale (POS)",
                 "E-commerce", "Sales & Marketing", "Personnel Management", "Financial Management",
                 "Analytics & Reporting"]:

        if choice == "Production Management":
            tab1, tab2, tab3 = st.tabs(["Production Tracking", "Workflow Management", "Product Formulations"])
            with tab1:
                production_management("Production Tracking")
            with tab2:
                production_management("Workflow Management")
            with tab3:
                production_management("Product Formulations")

        elif choice == "Inventory Management":
            tab1, tab2, tab3 = st.tabs(["Stock Levels", "Reorder Alerts", "Barcode Management"])
            with tab1:
                inventory_management("Stock Levels")
            with tab2:
                inventory_management("Reorder Alerts")
            with tab3:
                inventory_management("Barcode Management")

        elif choice == "Point of Sale (POS)":
            tab1, tab2, tab3 = st.tabs(["Process Sale", "View Sales History", "Void Sale"])
            with tab1:
                pos_module("Process Sale")
            with tab2:
                pos_module("View Sales History")
            with tab3:
                pos_module("Void Sale")

        elif choice == "E-commerce":
            tab1 = st.tabs(["Product Catalog"])
            with tab1[0]:
                ecommerce_module("Product Catalog")

        elif choice == "Sales & Marketing":
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Sales Performance", "Customer Insights", "Campaign Management",
                "Email Marketing", "Loyalty Programs", "Discount Management", "CRM"
            ])
            with tab1:
                sales_marketing("Sales Performance")
            with tab2:
                sales_marketing("Customer Insights")
            with tab3:
                sales_marketing("Campaign Management")
            with tab4:
                sales_marketing("Email Marketing")
            with tab5:
                sales_marketing("Loyalty Programs")
            with tab6:
                sales_marketing("Discount Management")
            with tab7:
                sales_marketing("CRM")

        elif choice == "Personnel Management":
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Employee Records", "Payroll Processing", "Attendance Tracking",
                "Performance Reviews", "Leave Management", "Shift Scheduling"
            ])
            with tab1:
                personnel_management("Employee Records")
            with tab2:
                personnel_management("Payroll Processing")
            with tab3:
                personnel_management("Attendance Tracking")
            with tab4:
                personnel_management("Performance Reviews")
            with tab5:
                personnel_management("Leave Management")
            with tab6:
                personnel_management("Shift Scheduling")

        elif choice == "Financial Management":
            tab1, tab2, tab3, tab4 = st.tabs(["Revenue Tracking", "Expense Tracking", "Tax Management", "Financial Reports"])
            with tab1:
                financial_management("Revenue Tracking")
            with tab2:
                financial_management("Expense Tracking")
            with tab3:
                financial_management("Tax Management")
            with tab4:
                financial_management("Financial Reports")

        elif choice == "Analytics & Reporting":
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Sales Analytics", "Customer Segmentation", "Inventory Analytics",
                "Financial Analytics", "Sales Forecasting", "Customer Lifetime Value (CLV)"
            ])
            with tab1:
                analytics_reporting("Sales Analytics")
            with tab2:
                analytics_reporting("Customer Segmentation")
            with tab3:
                analytics_reporting("Inventory Analytics")
            with tab4:
                analytics_reporting("Financial Analytics")
            with tab5:
                analytics_reporting("Sales Forecasting")
            with tab6:
                analytics_reporting("Customer Lifetime Value (CLV)")
    else:
        menu[choice]()

if __name__ == "__main__":
    main()