print ("python is way much better than c++")
import requests
import base64
import discord
import time
import mmap
import codecs
from discord.ext.commands import Bot
from discord.ext import commands
from bs4 import BeautifulSoup
page = requests.get("http://sareg.sci.cu.edu.eg/")
soup = BeautifulSoup(page.content, 'lxml')



data = page.content
with codecs.open('readme.html', 'wb', encoding="utf-8") as output:
    output.write(data.decode('utf-8'))
    

#with open ("tt.txt",'r' , encoding="utf-8")as file:
#    print(file.read())






PREFIX = ("+")
bot = commands.Bot(command_prefix=PREFIX, description='Hi')


class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        user=discord.user
        await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="\"It's Nice To Be Alive\"")) 
        await message.user('545269920640794627', "Your message")

    async def on_message(self, message):
        print('Message from {0.author}: {0.content}'.format(message))
        










































bot = MyClient()
bot.run('NzI0Njk1MTIyNTYxOTkwNzc2.XvD67Q.XMI-psY1IRGp9ypCygZCvjzAdlI')








#with open('sc.txt','r') as f:
 #   soup = BeautifulSoup(f.read(), "lxml")
  #  for line in soup.find_all('a'):
   #      print(line.text)


#with open('readme.txt', 'w') as f:
 #   f.write(str(page.content))


#base64_message = textwrap
#base64_bytes = base64_message.encode('ascii')
#message_bytes = base64.b64decode(base64_bytes)
#message = message_bytes.decode('ascii')
#print (page.content)