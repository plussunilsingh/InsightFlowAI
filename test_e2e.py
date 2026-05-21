from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py").run(timeout=30)
print("App loaded successfully!")

# Let's check if the search button exists
assert at.button[0].label == "🔍 Search"
assert at.button[1].label == "🛑 Stop Search"

# Input a query
at.text_input(key="query_input").input("Who is Sachin Tendulkar?").run()

# Click search
at.button[0].click().run(timeout=30)

print("Search completed!")
print("Response Output:")
for md in at.markdown:
    print(md.value)

print("\nAll Tests Passed Successfully!")
