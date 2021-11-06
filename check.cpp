#include <iostream>
using namespace std;
#include<fstream>
#include<vector>
#include <algorithm>
vector <string> outputs={};
vector <string> outputs_sec_loop={};
string output;
string output_sec_loop;
int lines_count;
int lines_count2=0;
int main (){
  system ("python the_scrper.py");
    fstream file;
    fstream file2;
    file.open("readme.html",ios::in);
    file2.open("sc.txt",ios::out);
      while (!file.eof()) {


         file >> output;
          char chars[] = "<>/\][}{;:''\"\n qweryuiop[]asdfgjkl;'zxcvbnm,./QWERTYUIOP{}ASDFGHJKL:ZXCVBNM~()-=html-_*&^$#@";

            for (int i = 0; i <= sizeof(chars); i++)
            {
                
                output.erase(remove(output.begin(), output.end(), chars[i]), output.end()); //remove A from string
            }

                if(output=="")
                    continue;
                outputs.push_back(output);
                file2<< output<<" ";
                if (lines_count % 12== 0)
                file2<< "\n";
                if(output=="التاريخ")
                 {
                  cout<< "found the news ....\n";
                    lines_count2 = lines_count;
                    break;
                  }
                lines_count++;
                
                
        }
                file2.close();
file2.open("tt.txt",ios::out);
int i =lines_count2 -27; 

  for (i ; i<= lines_count2;i++)
     {
 
      file2<<outputs[i]<<" ";
      
     }
        
      cout<<"sent to  the txt file ...\n" ;
                 file.close();
                 file2.close();
                 cout<<"done";
}