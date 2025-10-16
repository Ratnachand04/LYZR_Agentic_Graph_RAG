-- Create database
CREATE DATABASE REMA;

USE REMA;

-- Users table (common login for Sellers, Buyers, Agents, Admins)
CREATE TABLE Users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(15) UNIQUE NOT NULL,
    user_type VARCHAR(20) NOT NULL CHECK (user_type IN ('Seller', 'Buyer', 'Agent', 'Admin')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Properties table
CREATE TABLE Properties (
    property_id INT PRIMARY KEY AUTO_INCREMENT,
    seller_id INT NOT NULL,
    property_type VARCHAR(50) NOT NULL,
    size DECIMAL(10,2) NOT NULL,
    ownership VARCHAR(100) NOT NULL,
    features TEXT,
    description TEXT,
    estimated_price DECIMAL(15,2),
    total_value DECIMAL(15,2),
    future_value DECIMAL(15,2),
    seller_type VARCHAR(50) NOT NULL CHECK (seller_type IN ('Owner', 'Third Party', 'Agency')),
    negotiable BOOLEAN DEFAULT FALSE,
    brokering BOOLEAN DEFAULT FALSE,
    contact_info VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    usage_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'Available',
    FOREIGN KEY (seller_id) REFERENCES Users(user_id) ON DELETE CASCADE
);

-- Buyers table
CREATE TABLE Buyers (
    buyer_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    preferred_city VARCHAR(100),
    preferred_type VARCHAR(50),
    max_price DECIMAL(15,2),
    min_price DECIMAL(15,2),
    seller_type VARCHAR(50),
    interest_level VARCHAR(20) DEFAULT 'Medium',
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
);

-- Agents table
CREATE TABLE Agents (
    agent_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    business_type VARCHAR(50) NOT NULL,
    agency_name VARCHAR(100),
    license_number VARCHAR(50) UNIQUE,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
);

-- Transactions table
CREATE TABLE Transactions (
    transaction_id INT PRIMARY KEY AUTO_INCREMENT,
    property_id INT NOT NULL,
    buyer_id INT NOT NULL,
    agent_id INT,
    transaction_type VARCHAR(20) NOT NULL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    price DECIMAL(15,2) NOT NULL,
    payment_status VARCHAR(20) DEFAULT 'Pending',
    FOREIGN KEY (property_id) REFERENCES Properties(property_id) ON DELETE CASCADE,
    FOREIGN KEY (buyer_id) REFERENCES Buyers(buyer_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES Agents(agent_id) ON DELETE SET NULL
);

-- Contracts table
CREATE TABLE Contracts (
    contract_id INT PRIMARY KEY AUTO_INCREMENT,
    property_id INT NOT NULL,
    seller_id INT NOT NULL,
    buyer_id INT NOT NULL,
    contract_details TEXT NOT NULL,
    contract_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    contract_status VARCHAR(20) DEFAULT 'Active',
    FOREIGN KEY (property_id) REFERENCES Properties(property_id) ON DELETE CASCADE,
    FOREIGN KEY (seller_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (buyer_id) REFERENCES Buyers(buyer_id) ON DELETE CASCADE
);

-- Development Projects table
CREATE TABLE DevelopmentProjects (
    project_id INT PRIMARY KEY AUTO_INCREMENT,
    property_id INT NOT NULL,
    developer_id INT NOT NULL,
    project_details TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'Planned',
    estimated_completion_date DATE,
    FOREIGN KEY (property_id) REFERENCES Properties(property_id) ON DELETE CASCADE,
    FOREIGN KEY (developer_id) REFERENCES Agents(agent_id) ON DELETE CASCADE
);

-- Reviews table
CREATE TABLE Reviews (
    review_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    property_id INT NOT NULL,
    rating INT CHECK (rating BETWEEN 1 AND 5),
    review_text TEXT,
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (property_id) REFERENCES Properties(property_id) ON DELETE CASCADE
);

-- Payments table
CREATE TABLE Payments (
    payment_id INT PRIMARY KEY AUTO_INCREMENT,
    transaction_id INT NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    payment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES Transactions(transaction_id) ON DELETE CASCADE
);

-- Favorites table
CREATE TABLE Favorites (
    favorite_id INT PRIMARY KEY AUTO_INCREMENT,
    buyer_id INT NOT NULL,
    property_id INT NOT NULL,
    saved_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (buyer_id) REFERENCES Buyers(buyer_id) ON DELETE CASCADE,
    FOREIGN KEY (property_id) REFERENCES Properties(property_id) ON DELETE CASCADE
);

-- Notifications table
CREATE TABLE Notifications (
    notification_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
);

DROP DATABASE rema;