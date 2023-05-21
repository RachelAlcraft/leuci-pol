# Import Module
from tkinter import *
from tkhtmlview import HTMLLabel

# Create Object
root = Tk()

# Set Geometry
root.geometry("400x400")

# Add label
my_label = HTMLLabel(root, html="""
	<a href='https://www.geeksforgeeks.org/'>GEEKSFORGEEKS</a>
	

<p>Free Tutorials, Millions of Articles, Live, Online and Classroom Courses ,Frequent Coding Competitions ,Webinars by Industry Experts, Internship opportunities and Job Opportunities.</p>


	<img src="gfg.png">
	""")

# Adjust label
my_label.pack(pady=20, padx=20)

# Execute Tkinter
root.mainloop()
