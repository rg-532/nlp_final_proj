import os
import re
from datetime import datetime
from urllib.request import urlretrieve

import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

URI = "https://supreme.court.gov.il/Pages/fullsearch.aspx"

def scrape_supreme_court_verdicts(verbose: bool=True):
    save_dir = os.path.join(os.getcwd(), f"tmp_scv_corpus_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

    if verbose:
        print(f"""~~~ SCRAPE VERDICTS FROM SUPREME COURT WEBSITE ~~~
      Uses: Selenium + ChromeDriver
      Save location: {save_dir}\n\n""")

    # setup - access chrome driver and go to URI
    service = Service(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(URI)

    if verbose:
        print("Navigated to URI")

    # switch to relevant iframe
    form_frame = driver.find_element(By.XPATH, '//iframe')
    driver.switch_to.frame(form_frame)

    # find container for selecting document types
    # allow 10 second delay, since this element occasionally takes time loading in
    doc_type_select = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//sc-select[@ng-model="data.Type"]')),
        message="Err: Element Timeout"
    )

    # click text-field in container, to show dropdown
    doc_type_select.find_element(By.XPATH, './/input').click()

    # find option 'פסק-דין' to add it in
    option_verdicts = WebDriverWait(doc_type_select, 10).until(
        EC.visibility_of_element_located(
            (By.XPATH,'.//div[@class="ng-binding ng-scope" and text()="פסק-דין"]')
        ),message="Err: Element Timeout"
    )
    driver.execute_script("arguments[0].click()", option_verdicts)

    # click search button
    search_button = driver.find_element(
        By.XPATH, '//section[@class="search-bottom"]'
    ).find_element(
        By.XPATH, './/button[@type="submit"]'
    )
    driver.execute_script("arguments[0].click()", search_button)

    if verbose:
        print("Executed search for recent verdicts")
    

    # wait for results page to load
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.XPATH, '//form[@class="results-page ng-pristine ng-valid ng-scope ng-valid-pattern"]')
        ),
        message="Err: Page Switch Timeout"
    )
    
    # find scrollable container with results
    res_window = driver.find_element(By.XPATH, '//div[@class="results-listing"]')

    # get html button elements
    # scroll until the number of html links reaches 500 
    html_buttons = driver.find_elements(By.XPATH, '//a[@class="file-link html-link"]')

    while(len(html_buttons) < 500):
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight;',
                            res_window)
        html_buttons = driver.find_elements(By.XPATH, '//a[@class="file-link html-link"]')
    
    # get urls to htmls
    html_hrefs = [e.get_attribute('href') for e in html_buttons]
    html_links = [h if h.startswith("https://supremedecisions.court.gov.il")
                 else f"https://supremedecisions.court.gov.il/{h}"
                 for h in html_hrefs]

    if verbose:
        print(f"Retrieved {len(html_links)} html links")
    
    driver.close()

    os.mkdir(save_dir)

    for link in html_links:
        filename = f"{re.findall(r'fileName=.*&', link)[0][9:-1]}.html"
        urlretrieve(link, f"{save_dir}/{filename}")

    if verbose:
        print(f"Saved {len(html_links)} html files to {save_dir}")
        print("Done!")


scrape_supreme_court_verdicts()