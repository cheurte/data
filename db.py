import pyodbc
import argparse

class SQLcust():
    r'''
    Classe pour pouvoir se connecter et agir plus facilement sur
    les bases de données.
    '''
    def   __init__ (self,
                    server   = '192.168.7.53',
                    uid      = 'corentin',
                    pwd      = 'Biotec1234!',
                    port     = 4430,
                    column_names = '../all_column_prod.txt',
                    save_location = '../dataTest/',
                    product_path= '../Enermeter/')->None:
        r'''
        Class to make the relationship between quality and production, and to download the production data a the right moment.

        # Parameters :

        server :
            IP adress of the server. Default : 192.168.7.53
            type [str]

        uid :
            username of the connection. Default : corentin
            type [str]

        '''
        self.server = server
        self.uid    = uid
        self.pwd    = pwd
        self.port   = port

        self.col_names = column_names
        self.save_loc = save_location
        self.product_path = product_path

        self.enemeter = None
        self.csnr = None
        self.quality = None

        self.connectorQS = self.conn('QS_Data')
        self.connectorMess = self.conn('Messwarte')

    def conn(self, database):
        return  pyodbc.connect(
            f"DRIVER=cubeSQL ODBC;\
            Server={self.server};port={self.port};uid={self.uid};pwd={self.pwd};Database={database};",autocommit=False)

    def close(self):
        '''
        Permet de fermer le connecteur
        '''
        self.connectorQS.close()
        self.connectorMess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser("manage quality data")
    # parser.add_argument("--config", "-c", default="C:\\Users\\corentin.heurte\\Documents\\data\\config\\config.json")
    # args = parser.parse_args()
    conn = SQLcust()
