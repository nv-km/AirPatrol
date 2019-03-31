# Python code to illustrate Sending mail from 
# your Gmail account 
import smtplib 

def mail():
	# creates SMTP session 
	s = smtplib.SMTP('smtp.gmail.com', 587) 

	# start TLS for security 
	s.starttls() 

	# Authentication 
	s.login("deepbluepwp16@gmail.com", "Deepblue17") 

	# message to be sent 
	message = "Warning Deforestation has been detected, please deploy drone to verify."

	# sending the mail 
	s.sendmail("deepbluepwp16@gmail.com", "apurva.mhatre16@siesgst.ac.in", message) 
	print("Mail sent")
	# terminating the session 
	s.quit()