

import numpy as np
import pandas as pd
import re
from astropy.coordinates import SkyCoord
from pandas.api.types import is_numeric_dtype
import requests
from bs4 import BeautifulSoup
# import html5lib


class HyperLedaTable:
    def __init__(self, gal_name):
        '''
        Used to retrieve galaxy meta data from Hyperleda

        Attributes
        ----------
        gal_name : str
            the name of the galaxy whos parameters are to be obtained
        
        table : pd.DataFrame (optional)
            data frame of all parameters that have been collected, will only be initialised after first
            get_table call and is updated (but not replaced) with new information when subsequent get_table
            calls are made
        '''
        self.gal_name = gal_name # galaxy name that all data will be for

    def get_table(self, url_type, url=None, drop_columns = ['Description', 'method', 'calib', 'L'], select_criteria = None):
        ''' 
        Load table from Hyperleda and update self.table to append new table (or create table if no existing table)

        Parameters
        ----------
        url_type : str
            either 'main', 'pa', 'distance', 'incl' or 'ra_dec'
        
        url : str
            a hyperleda url (optional), default behaviour is to use the url_type and self.get_url
            if url is provided should end in "&a=htab" (write something to fix this for generalisation)

        drop_columns : list
            columns to drop from the table, if undropped they become parameters (except for source type columns)

        select_criteria : dict
            values of parameters to satisfy in row selection

            
        Returns
        -------
        table : pd.DataFrame
            data frame of parameters loaded
        

        '''
        if url is None:
            url = self.get_url(url_type)

        if url_type == 'ra_dec':
            table = self.get_ra_dec()


        else:

            if url_type == 'main':

                table = pd.read_html(url, match='objtype')[0]

            else:
                table = pd.read_html(url)[0]
        


            table.drop(drop_columns, axis='columns', errors='ignore', inplace=True)
            if len(table) == 0:
                raise Exception('No Table Found.')
            
            if len(table) > 1:

                print(f'Warning: Multiple values of {url_type} found for {self.gal_name}, using first complete row \n url: {url}')
                
                if select_criteria is not None:
                    table = self.check_search_criteria(table, select_criteria)

   
                table = table.iloc[0].to_frame().T
                

            table = self.extract_source(table)
            try:
                table.columns = [col.replace('_', '') for col in table.columns]

            except AttributeError:
                pass

            table = pd.melt(table, id_vars=['source'], value_vars=table.columns[table.columns!='source'], var_name='Parameter', value_name='Value')
         
            table = table.apply(self.seperate_uncertainty, axis=1, result_type='expand')


        try: # try to add the current table to existing table and if not then set the table attribute with current table
            self.table = pd.concat([self.table, table], ignore_index=True)
        except:        

            setattr(self, 'table', table)


        

        # drop duplicates and nans
        self.table.dropna(subset='Value', axis=0, how='any', inplace=True)
        self.table.drop_duplicates(subset='Parameter', inplace=True)


        return table
    
    def find_in_table(self, keys, table=None, set_attr=True):

        ''' 
        find parameters in table and set as attributes

        Parameters
        ----------
            keys : list
                list of parameters to find and return info
            
            table : pd.DataFrame (optional)
                table to search through if None, use table attribute
            
            set_attr : bool (optional)
                whether to set the search keys as attributes, default is True

            
            
        '''

        if table is None:
            table = self.table

        

        param_dict = self.table_to_dict(table[table['Parameter'].isin(keys)])


        if set_attr:
            try:
                self.info_dict
            except:
                setattr(self, 'info_dict', {})
            for key in keys:
                # if key in param
                if key in param_dict.keys():
                    setattr(self, key, param_dict[key]['Value'])

                    self.info_dict[key] = param_dict[key]



        return param_dict
    
    def check_search_criteria(self, table, search_criteria):
        ''' 
        Apply search criteria to select row of table given other parameters


        Parameters
        ----------
        table : pd.DataFramwe
            table to select row from
        
        search_criteria : dict
            parameter keys and values to select data by
        
        Return
        ------
        selected_rows : pd.DataFrame
            table with selected row

        '''
        mask = np.ones(len(table))


        for col_name, val in search_criteria.items():
                    

            mask *= (table[col_name] == val)
        

        return table[mask.astype('bool')]

    def string_to_float(self, numeric_string):
        ''' 
        convert a string into a float if possible otherwise return the string

        Parameters
        ----------
        numeric_string : str
            string to attempt to convert
        
        Returns
        -------
        value_as_float : float or str
            the result of the conversion
        '''
        try:
            return float(numeric_string)
        except ValueError:
            return numeric_string
    

    def seperate_uncertainty(self, row):
        ''' 
        Turn a data string into (value, uncertainty)

        Parameters
        ----------
        value_uncertainty : str
            string which may or maynot contain an uncertainty (indicated by ±)
            if contains value for uncertainty, the two are seperated and returned as a tuple of floats
            if no uncertainty, the uncertainty value 'NA' is returned
            if the value is numeric it is returned as float
        
        Returns
        -------
        value : float or str
            the uncertainty separated value. if numeric, value is a float.
        
        uncertainty : float or str
            the uncertainty after separation. if no separation, 'NA' is returned

        '''
        value_uncertainty = str(row['Value']).replace(u'\xa0', '')
        spl = value_uncertainty.split('±')

        if len(spl) == 1:
            spl = spl[0].split('(')
            row['Uncertainty'] = 'NA'
        else:
            row['Uncertainty'] = self.string_to_float(spl[1])
        
        
        row['Value'] = self.string_to_float(spl[0])

        return row
    
    def table_to_dict(self, table=None):
        '''turn table into dictionary with correct formatting'''
        if table is None:
            table = self.table



        table.index = table['Parameter']



        return table.to_dict(orient='index')
    
    def get_source_by_id(self, source_id):
        '''get the citation from an iref or dataset id number'''
        source_url = f'http://atlas.obs-hp.fr/hyperleda/B.cgi?n=a103&b={source_id}'

        source_df = pd.read_html(source_url)[0]

        return source_df.at[0,3]
    


    
    def extract_source(self, table):
        '''
        find sources in table and create new column containing, if no source for parameter in table, use hyperleda instead
        '''
        if 'bibref' in table.keys():
            sources = table['bibref']

            
        
        elif 'iref' in table.keys():
            sources = table['iref']
        
            sources = sources.apply(self.get_source_by_id)

        elif 'dataset' in table.keys():
            sources = table['dataset']

            sources = sources.apply(self.get_source_by_id)




        else:
            sources = 'hyperleda'


        table.drop(['bibref', 'iref', 'dataset'], axis='columns', errors='ignore', inplace=True)

        table['source'] = sources
        return table
    

    def get_url(self, url_type, gal_name = None, url_dict = None):
        '''
        get the url for a dataset from the url_type

        allowed types at the moment:
            - 'main': for the main object table
            - 'pa' : for the pa table
            - 'incl' : Diameters
            - 'distance' : for the distances table
            - 'ra_dec' : gives you the home page but don't really need it anymore? [delete ???]

        '''
        if gal_name is None:
            gal_name = self.gal_name
        if url_dict is None:
            url_dict = {
                'main': f'http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?{gal_name}&a=htab',
                'pa':  f"http://atlas.obs-hp.fr/hyperleda/fG.cgi?n=a103&c=o&of=1,leda,simbad&nra=l&o={gal_name}&a=htab",
                'incl': f"http://atlas.obs-hp.fr/hyperleda/fG.cgi?n=a106&c=o&of=1,leda,simbad&nra=l&o={gal_name}&a=htab",  #incl
                'distance': f"http://atlas.obs-hp.fr/hyperleda/fG.cgi?n=a007&c=o&of=1,leda,simbad&nra=l&o={gal_name}&a=htab",
                'ra_dec': f'http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?{gal_name}'
            }

        return url_dict[url_type]
    
    def parameters_dict(self, info=True):
        '''
        get all stored parameters as dictionary (excluding table)
        '''

        param_dict = self.__dict__.copy()

        param_dict.pop('table')

        info = param_dict.pop('info_dict')
        
        return param_dict, info


    def get_ra_dec(self):
        '''
        get RA DEC from gal_name using astropy

        '''
        coords = SkyCoord.from_name(self.gal_name)
        table = pd.DataFrame({'Parameter': ['RA', 'DEC'], 'Value': [coords.ra.deg, coords.dec.deg]})

        table['Uncertainty'] = 'NA'
        table['source'] = 'astropy'
        return table

        


    
    
    
    

