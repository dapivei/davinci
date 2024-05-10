import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def scrape_article_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all links with class "entry-title"
        links = soup.find_all("a", class_="entry-title")

        # Extract href and title from each link
        article_links = [{"title": link.get_text(), "href": link.get("href")} for link in links]

        return article_links

    except requests.exceptions.RequestException as e:
        print("Error fetching content:", e)
        return None

# Base URL for articles
base_url = "https://lopezobrador.org.mx/"

# Start and end dates for scraping articles
start_date = datetime(2023, 3, 18)
end_date = datetime(2023, 3, 19)  # Change this to the desired end date

# Initialize empty list to store all article links
all_article_links = []

# Loop over dates and scrape articles
current_date = start_date
while current_date <= end_date:
    formatted_date = current_date.strftime("%Y/%m/%d")
    url = base_url + formatted_date + "/"
    print("Scraping articles from:", url)

    # Scrape article links from the current date
    article_links = scrape_article_links(url)
    if article_links:
        all_article_links.extend(article_links)
    else:
        print("Failed to fetch articles for:", url)

    # Move to the next date
    current_date += timedelta(days=1)

# Save the data into a JSON file
output_file = "article_links.json"
with open(output_file, "w") as f:
    json.dump(all_article_links, f, indent=4)

print("Article links saved to:", output_file)

