import sqlite3
import os

def create_views(db_path="data/northwind.sqlite"):
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Waiting for download...")
        return

    print(f"Connecting to {db_path} to add views...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List of views in dependency order
    views_sql = [
        """DROP VIEW IF EXISTS "Alphabetical list of products";""",
        """CREATE VIEW "Alphabetical list of products" AS
           SELECT Products.*, Categories.CategoryName
           FROM Categories 
           INNER JOIN Products ON Categories.CategoryID = Products.CategoryID
           WHERE (((Products.Discontinued)=0));""",

        """DROP VIEW IF EXISTS "Current Product List";""",
        """CREATE VIEW "Current Product List" AS
           SELECT ProductID, ProductName FROM Products WHERE Discontinued=0;""",

        """DROP VIEW IF EXISTS "Customer and Suppliers by City";""",
        """CREATE VIEW "Customer and Suppliers by City" AS
           SELECT City, CompanyName, ContactName, 'Customers' AS Relationship FROM Customers
           UNION 
           SELECT City, CompanyName, ContactName, 'Suppliers' FROM Suppliers 
           ORDER BY City, CompanyName;""",

        """DROP VIEW IF EXISTS "Invoices";""",
        """CREATE VIEW "Invoices" AS
           SELECT Orders.ShipName, Orders.ShipAddress, Orders.ShipCity, Orders.ShipRegion, 
           Orders.ShipPostalCode, Orders.ShipCountry, Orders.CustomerID, Customers.CompanyName AS CustomerName, 
           Customers.Address, Customers.City, Customers.Region, Customers.PostalCode, Customers.Country,
           (Employees.FirstName + ' ' + Employees.LastName) AS Salesperson, 
           Orders.OrderID, Orders.OrderDate, Orders.RequiredDate, Orders.ShippedDate, 
           Shippers.CompanyName As ShipperName, "Order Details".ProductID, Products.ProductName, 
           "Order Details".UnitPrice, "Order Details".Quantity, "Order Details".Discount, 
           ((("Order Details".UnitPrice*Quantity*(1-Discount))/100)*100) AS ExtendedPrice, Orders.Freight 
           FROM Customers 
           JOIN Orders ON Customers.CustomerID = Orders.CustomerID  
           JOIN Employees ON Employees.EmployeeID = Orders.EmployeeID    
           JOIN "Order Details" ON Orders.OrderID = "Order Details".OrderID     
           JOIN Products ON Products.ProductID = "Order Details".ProductID      
           JOIN Shippers ON Shippers.ShipperID = Orders.ShipVia;""",

        """DROP VIEW IF EXISTS "Orders Qry";""",
        """CREATE VIEW "Orders Qry" AS
           SELECT Orders.OrderID, Orders.CustomerID, Orders.EmployeeID, Orders.OrderDate, 
           Orders.RequiredDate, Orders.ShippedDate, Orders.ShipVia, Orders.Freight,
           Orders.ShipName, Orders.ShipAddress, Orders.ShipCity, Orders.ShipRegion,
           Orders.ShipPostalCode, Orders.ShipCountry, Customers.CompanyName, Customers.Address,
           Customers.City, Customers.Region, Customers.PostalCode, Customers.Country
           FROM Customers 
           JOIN Orders ON Customers.CustomerID = Orders.CustomerID;""",

        """DROP VIEW IF EXISTS "Order Subtotals";""",
        """CREATE VIEW "Order Subtotals" AS
           SELECT "Order Details".OrderID, 
           Sum(("Order Details".UnitPrice*Quantity*(1-Discount)/100)*100) AS Subtotal
           FROM "Order Details"
           GROUP BY "Order Details".OrderID;""",

        """DROP VIEW IF EXISTS "Product Sales for 1997";""",
        """CREATE VIEW "Product Sales for 1997" AS
           SELECT Categories.CategoryName, Products.ProductName, 
           Sum(("Order Details".UnitPrice*Quantity*(1-Discount)/100)*100) AS ProductSales
           FROM Categories
           JOIN Products On Categories.CategoryID = Products.CategoryID
           JOIN "Order Details" on Products.ProductID = "Order Details".ProductID     
           JOIN "Orders" on Orders.OrderID = "Order Details".OrderID 
           WHERE Orders.ShippedDate Between DATE('1997-01-01') And DATE('1997-12-31')
           GROUP BY Categories.CategoryName, Products.ProductName;""",

        """DROP VIEW IF EXISTS "Products Above Average Price";""",
        """CREATE VIEW "Products Above Average Price" AS
           SELECT Products.ProductName, Products.UnitPrice
           FROM Products
           WHERE Products.UnitPrice>(SELECT AVG(UnitPrice) From Products);""",

        """DROP VIEW IF EXISTS "Products by Category";""",
        """CREATE VIEW "Products by Category" AS
           SELECT Categories.CategoryName, Products.ProductName, Products.QuantityPerUnit, 
           Products.UnitsInStock, Products.Discontinued
           FROM Categories 
           INNER JOIN Products ON Categories.CategoryID = Products.CategoryID
           WHERE Products.Discontinued <> 1;""",

        """DROP VIEW IF EXISTS "Quarterly Orders";""",
        """CREATE VIEW "Quarterly Orders" AS
           SELECT DISTINCT Customers.CustomerID, Customers.CompanyName, Customers.City, Customers.Country
           FROM Customers 
           JOIN Orders ON Customers.CustomerID = Orders.CustomerID
           WHERE Orders.OrderDate BETWEEN DATE('1997-01-01') And DATE('1997-12-31');""",

        """DROP VIEW IF EXISTS "Sales Totals by Amount";""",
        """CREATE VIEW "Sales Totals by Amount" AS
           SELECT "Order Subtotals".Subtotal AS SaleAmount, Orders.OrderID, 
           Customers.CompanyName, Orders.ShippedDate
           FROM Customers 
           JOIN Orders ON Customers.CustomerID = Orders.CustomerID
           JOIN "Order Subtotals" ON Orders.OrderID = "Order Subtotals".OrderID 
           WHERE ("Order Subtotals".Subtotal >2500) 
           AND (Orders.ShippedDate BETWEEN DATE('1997-01-01') And DATE('1997-12-31'));""",

        """DROP VIEW IF EXISTS "Summary of Sales by Quarter";""",
        """CREATE VIEW "Summary of Sales by Quarter" AS
           SELECT Orders.ShippedDate, Orders.OrderID, "Order Subtotals".Subtotal
           FROM Orders 
           INNER JOIN "Order Subtotals" ON Orders.OrderID = "Order Subtotals".OrderID
           WHERE Orders.ShippedDate IS NOT NULL;""",

        """DROP VIEW IF EXISTS "Summary of Sales by Year";""",
        """CREATE VIEW "Summary of Sales by Year" AS
           SELECT Orders.ShippedDate, Orders.OrderID, "Order Subtotals".Subtotal
           FROM Orders 
           INNER JOIN "Order Subtotals" ON Orders.OrderID = "Order Subtotals".OrderID
           WHERE Orders.ShippedDate IS NOT NULL;""",

        """DROP VIEW IF EXISTS "Category Sales for 1997";""",
        """CREATE VIEW "Category Sales for 1997" AS
           SELECT "Product Sales for 1997".CategoryName, 
           Sum("Product Sales for 1997".ProductSales) AS CategorySales
           FROM "Product Sales for 1997"
           GROUP BY "Product Sales for 1997".CategoryName;""",

        """DROP VIEW IF EXISTS "Order Details Extended";""",
        """CREATE VIEW "Order Details Extended" AS
           SELECT "Order Details".OrderID, "Order Details".ProductID, Products.ProductName, 
           "Order Details".UnitPrice, "Order Details".Quantity, "Order Details".Discount, 
           ("Order Details".UnitPrice*Quantity*(1-Discount)/100)*100 AS ExtendedPrice
           FROM Products 
           JOIN "Order Details" ON Products.ProductID = "Order Details".ProductID;""",

        """DROP VIEW IF EXISTS "Sales by Category";""",
        """CREATE VIEW "Sales by Category" AS
           SELECT Categories.CategoryID, Categories.CategoryName, Products.ProductName, 
           Sum("Order Details Extended".ExtendedPrice) AS ProductSales
           FROM Categories 
           JOIN Products ON Categories.CategoryID = Products.CategoryID
           JOIN "Order Details Extended" ON Products.ProductID = "Order Details Extended".ProductID                
           JOIN Orders ON Orders.OrderID = "Order Details Extended".OrderID 
           WHERE Orders.OrderDate BETWEEN DATE('1997-01-01') And DATE('1997-12-31')
           GROUP BY Categories.CategoryID, Categories.CategoryName, Products.ProductName;"""
    ]

    for sql in views_sql:
        try:
            cursor.execute(sql)
            print(f"Executed view creation: {sql.splitlines()[0]}")
        except Exception as e:
            print(f"Error executing view: {e}")

    conn.commit()
    conn.close()
    print("✅ All views added successfully.")

if __name__ == "__main__":
    create_views()