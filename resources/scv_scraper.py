import selenium
from selenium import webdriver

URI = "https://supreme.court.gov.il/Pages/fullsearch.aspx"

driver = webdriver.Chrome()
driver.get(URI)
