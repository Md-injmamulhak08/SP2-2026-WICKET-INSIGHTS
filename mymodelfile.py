import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class MyModel:
    """
    Sona Gameathon 2026 - Team Wicket Insights
    """
    def __init__(self):
        lgb_params = {
            'n_estimators': 300,        
            'learning_rate': 0.04,      
            'max_depth': 7,            
            'num_leaves': 40,          
            'min_child_samples': 10,   
            'subsample': 0.8,           
            'colsample_bytree': 0.8,   
            'random_state': 42,
            'n_jobs': 1,
            'verbose': -1
        }
        self.model_inn1 = lgb.LGBMRegressor(**lgb_params)
        self.model_inn2 = lgb.LGBMRegressor(**lgb_params)
        
        self.team_encoder = LabelEncoder()
        self.venue_encoder = LabelEncoder()
        
        self.mean_score = 0.0
        self.global_pp_econ = 8.5 
        self.mean_batters = 2 
        self.modern_target = 0.0
        self.old_target = 0.0
        
        self.known_teams = set()
        self.known_venues = set()
        
        self.player_sr_dict = {}     
        self.player_econ_dict = {}   
        self.player_id_map = {} 
        self.venue_form_dict = {}
        self.h2h_dict = {} 

    def _standardize_columns(self, df):
        """Forces dirty hackathon columns into standard names without creating duplicates."""
        col_map = {
            'City': 'venue', 'city': 'venue', 'stadium': 'venue',
            'BattingTeam': 'batting_team',
            'BowlingTeam': 'bowling_team'
        }
        
        for dirty, clean in col_map.items():
            if dirty in df.columns:
                if clean not in df.columns:
                    df = df.rename(columns={dirty: clean})
                else:
                    # If both exist, merge them to avoid NaN holes, then drop the dirty one
                    df[clean] = df[clean].fillna(df[dirty])
                    df = df.drop(columns=[dirty])
                    
        # Ultimate Duplicate Column Killer 
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _extract_year(self, season_val):
        try:
            match = re.search(r'\d{4}', str(season_val))
            return int(match.group()) if match else 2026
        except:
            return 2026

    def _clean_text(self, text):
        if not isinstance(text, str): return "unknown"
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        fluff = r'\b(stadium|cricket|international|association|pca|intl|ground|sports|academy|complex|arena|club)\b'
        return ' '.join(re.sub(fluff, '', text).split())

    def fit(self, ball_data, player_info=None, match_info=None):
        ball_data = ball_data.copy()
        ball_data = self._standardize_columns(ball_data)
        
        m_id, v_col, date_col = 'id', 'venue', None
        if match_info is not None:
            match_info = match_info.copy()
            match_info = self._standardize_columns(match_info)
            m_id = next((c for c in ['id', 'ID', 'matchId'] if c in match_info.columns), 'id')
            v_col = next((c for c in ['venue', 'Venue'] if c in match_info.columns), 'venue')
            date_col = next((c for c in ['date', 'season', 'year'] if c in match_info.columns), None)
            
        if player_info is not None:
            id_col = next((c for c in ['ID', 'player_id', 'id', 'identifier'] if c in player_info.columns), None)
            if id_col:
                name_col = next((c for c in ['Player_Name', 'player_name', 'name', 'FullName'] if c in player_info.columns), player_info.columns[1] if len(player_info.columns) > 1 else id_col)
                self.player_id_map = dict(zip(player_info[name_col].astype(str).str.lower().str.strip(), player_info[id_col].astype(str)))

        if 'innings' in ball_data.columns and 'inning' not in ball_data.columns:
            ball_data.rename(columns={'innings': 'inning'}, inplace=True)
            
        pp_balls = ball_data[ball_data['over'] < 6].copy()
        bat_col = next((c for c in ['batsman_id', 'striker_id', 'batter_id', 'batsman', 'striker', 'batter'] if c in pp_balls.columns), 'batsman')
        bowl_col = next((c for c in ['bowler_id', 'bowler', 'bowler_name'] if c in pp_balls.columns), 'bowler')
        
        pp_balls['total_runs'] = pp_balls.get('batsman_runs', pd.Series(0, index=pp_balls.index)).fillna(0) + \
                                 pp_balls.get('extras', pd.Series(0, index=pp_balls.index)).fillna(0)
                                 
        if match_info is not None and date_col:
            date_map = dict(zip(match_info[m_id].astype(str), match_info[date_col].astype(str)))
            pp_balls['sort_key'] = pp_balls['matchId'].astype(str).map(date_map).fillna(pp_balls['matchId'].astype(str))
        else:
            pp_balls['sort_key'] = pp_balls['matchId']
            
        # Batter Sliding Window
        player_stats = pp_balls.groupby([bat_col, 'matchId', 'sort_key']).agg(
            runs=('batsman_runs', 'sum'), balls=(bat_col, 'count')
        ).reset_index().sort_values(by=[bat_col, 'sort_key', 'matchId'])
        
        player_stats['rolling_runs'] = player_stats.groupby(bat_col)['runs'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        player_stats['rolling_balls'] = player_stats.groupby(bat_col)['balls'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        player_stats['recent_sr'] = np.where(player_stats['rolling_balls'] > 0, (player_stats['rolling_runs'] / player_stats['rolling_balls']) * 100, np.nan)
        
        latest_sr = player_stats.dropna(subset=['recent_sr']).drop_duplicates(subset=[bat_col], keep='last')
        self.player_sr_dict = dict(zip(latest_sr[bat_col].astype(str), latest_sr['recent_sr']))
        
        pp_balls = pp_balls.merge(player_stats[[bat_col, 'matchId', 'recent_sr']], on=[bat_col, 'matchId'], how='left')
        pp_balls['recent_sr'] = pp_balls['recent_sr'].fillna(145.0) 

        # Bowler Sliding Window
        bowler_stats = pp_balls.groupby([bowl_col, 'matchId', 'sort_key']).agg(
            runs_conceded=('total_runs', 'sum'), balls_bowled=(bowl_col, 'count')
        ).reset_index().sort_values(by=[bowl_col, 'sort_key', 'matchId'])
        
        bowler_stats['rolling_runs'] = bowler_stats.groupby(bowl_col)['runs_conceded'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        bowler_stats['rolling_balls'] = bowler_stats.groupby(bowl_col)['balls_bowled'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        bowler_stats['recent_econ'] = np.where(bowler_stats['rolling_balls'] > 0, (bowler_stats['rolling_runs'] / bowler_stats['rolling_balls']) * 6, np.nan)
        
        latest_econ = bowler_stats.dropna(subset=['recent_econ']).drop_duplicates(subset=[bowl_col], keep='last')
        self.player_econ_dict = dict(zip(latest_econ[bowl_col].astype(str), latest_econ['recent_econ']))
        
        pp_balls = pp_balls.merge(bowler_stats[[bowl_col, 'matchId', 'recent_econ']], on=[bowl_col, 'matchId'], how='left')
        
        mean_econ = pp_balls['recent_econ'].mean()
        self.global_pp_econ = mean_econ if pd.notna(mean_econ) else 8.5
        pp_balls['recent_econ'] = pp_balls['recent_econ'].fillna(self.global_pp_econ)

        batters_count = pp_balls.groupby(['matchId', 'inning'])[bat_col].nunique().reset_index(name='num_batters')

        pp_totals = pp_balls.groupby(['matchId', 'inning', 'batting_team', 'bowling_team'])['total_runs'].sum().reset_index()
        pp_totals = pp_totals.merge(batters_count, on=['matchId', 'inning'], how='left')
        
        mean_bats = pp_totals['num_batters'].mean()
        self.mean_batters = int(round(mean_bats)) if pd.notna(mean_bats) else 2
        
        sort_key_map = dict(zip(pp_balls['matchId'], pp_balls['sort_key']))
        pp_totals['sort_key'] = pp_totals['matchId'].map(sort_key_map)
        
        match_stats = pp_balls.groupby(['matchId', 'inning']).agg(
            top_order_sr=('recent_sr', 'mean'), top_bowler_econ=('recent_econ', 'mean')
        ).reset_index()
        pp_totals = pp_totals.merge(match_stats, on=['matchId', 'inning'], how='left')
        
        b_runs = ball_data.get('batsman_runs', pd.Series(0, index=ball_data.index)).fillna(0)
        e_runs = ball_data.get('extras', pd.Series(0, index=ball_data.index)).fillna(0)
        temp_df = ball_data[['matchId', 'inning']].copy()
        temp_df['total_ball_runs'] = b_runs + e_runs
        first_inn = temp_df[temp_df['inning'] == 1].groupby('matchId')['total_ball_runs'].sum().reset_index(name='target_score')
        
        pp_totals = pp_totals.merge(first_inn, on='matchId', how='left')
        
        if match_info is not None:
            venues = match_info[[m_id, v_col]].copy()
            venues.rename(columns={m_id: 'matchId', v_col: 'venue'}, inplace=True)
            venues['year'] = match_info[date_col].apply(self._extract_year) if date_col else 2026
            pp_totals = pp_totals.merge(venues, on='matchId', how='left')
        else:
            pp_totals['venue'], pp_totals['year'] = 'unknown', 2026

        self.mean_score = pp_totals['total_runs'].mean() if not pp_totals.empty else 55.0

        pp_totals['batting_team'] = pp_totals['batting_team'].apply(self._clean_text)
        pp_totals['bowling_team'] = pp_totals['bowling_team'].apply(self._clean_text)
        pp_totals['venue'] = pp_totals['venue'].apply(self._clean_text)

        pp_totals = pp_totals.sort_values(by=['batting_team', 'bowling_team', 'inning', 'sort_key', 'matchId'])
        pp_totals['h2h_avg'] = pp_totals.groupby(['batting_team', 'bowling_team', 'inning'])['total_runs'].transform(lambda x: x.shift(1).expanding().mean())
        pp_totals['h2h_avg'] = pp_totals['h2h_avg'].fillna(self.mean_score)
        
        self.h2h_dict = pp_totals.groupby(['batting_team', 'bowling_team', 'inning'])['total_runs'].mean().to_dict()

        pp_totals = pp_totals.sort_values(by=['venue', 'inning', 'sort_key', 'matchId'])
        pp_totals['recent_venue_avg'] = pp_totals.groupby(['venue', 'inning'])['total_runs'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        
        latest_venue = pp_totals.dropna(subset=['recent_venue_avg']).drop_duplicates(subset=['venue', 'inning'], keep='last')
        self.venue_form_dict = dict(zip(zip(latest_venue['venue'], latest_venue['inning']), latest_venue['recent_venue_avg']))
        pp_totals['recent_venue_avg'] = pp_totals['recent_venue_avg'].fillna(self.mean_score if self.mean_score > 0 else 62.0)

        inn_1_data = pp_totals[pp_totals['inning'] == 1]
        self.modern_target = inn_1_data[inn_1_data['year'] >= 2023]['target_score'].mean()
        if pd.isna(self.modern_target): self.modern_target = 185.0
        self.old_target = inn_1_data[inn_1_data['year'] < 2023]['target_score'].mean()
        if pd.isna(self.old_target): self.old_target = 165.0

        pp_totals.loc[pp_totals['inning'] == 1, 'target_score'] = 0.0
        pp_totals['target_score'] = pp_totals['target_score'].fillna(0)
            
        self.known_teams.update(pp_totals['batting_team'].unique())
        self.known_teams.update(pp_totals['bowling_team'].unique())
        self.known_venues.update(pp_totals['venue'].unique())
        
        self.team_encoder.fit(sorted(list(self.known_teams) + ['unknown']))
        self.venue_encoder.fit(sorted(list(self.known_venues) + ['unknown']))
        
        pp_totals['bat_encoded'] = self.team_encoder.transform(pp_totals['batting_team'])
        pp_totals['bowl_encoded'] = self.team_encoder.transform(pp_totals['bowling_team'])
        pp_totals['venue_encoded'] = self.venue_encoder.transform(pp_totals['venue'])
        pp_totals['is_modern'] = (pp_totals['year'] >= 2023).astype(int)
        
        self.features_inn1 = ['bat_encoded', 'bowl_encoded', 'venue_encoded', 'year', 'is_modern', 
                              'top_order_sr', 'top_bowler_econ', 'recent_venue_avg', 'num_batters', 'h2h_avg']
        self.features_inn2 = self.features_inn1 + ['target_score']
        
        inn1_df = pp_totals[pp_totals['inning'] == 1]
        inn2_df = pp_totals[pp_totals['inning'] == 2]
        
        if not inn1_df.empty:
            self.model_inn1.fit(inn1_df[self.features_inn1], inn1_df['total_runs'], categorical_feature=['bat_encoded', 'bowl_encoded', 'venue_encoded'])
        if not inn2_df.empty:
            self.model_inn2.fit(inn2_df[self.features_inn2], inn2_df['total_runs'], categorical_feature=['bat_encoded', 'bowl_encoded', 'venue_encoded'])
            
        return self

    def predict(self, test_df):
        test_df = test_df.copy()
        test_df = self._standardize_columns(test_df)
        
        predictions = []
        for _, row in test_df.iterrows():
            t_bat = self._clean_text(row.get('batting_team', 'unknown'))
            t_bowl = self._clean_text(row.get('bowling_team', 'unknown'))
            v = self._clean_text(row.get('venue', 'unknown'))
            inn = int(row.get('innings', row.get('inning', 1)))
            y = self._extract_year(row.get('season', row.get('date', 2026))) 
            
            t_bat = t_bat if t_bat in self.known_teams else 'unknown'
            t_bowl = t_bowl if t_bowl in self.known_teams else 'unknown'
            v = v if v in self.known_venues else 'unknown'
            
            bat_enc = self.team_encoder.transform([t_bat]).item()
            bowl_enc = self.team_encoder.transform([t_bowl]).item()
            ven_enc = self.venue_encoder.transform([v]).item()
            is_modern = 1 if y >= 2023 else 0
            
            target = 0.0
            target_found = False
            if inn == 2:
                for col_name, val in row.items():
                    if any(kw in str(col_name).lower() for kw in ['target', '1_run', 'inn1', 'first_inn', 'score_1']):
                        try:
                            if float(val) > 20:  
                                target = float(val)
                                target_found = True
                                break
                        except ValueError: continue
                if not target_found: 
                    target = self.modern_target if is_modern else self.old_target
            
            bat_keys = []
            bowl_keys = []
            for bat_col_name in ['batsman_id', 'striker_id', 'batter_id', 'batsman_1_id', 'batsman_2_id', 'batsman', 'striker', 'batter', 'batsmen', 'batting_players']:
                val = row.get(bat_col_name)
                if val is not None and str(val).strip().lower() not in ['nan', 'none', '', 'na', 'null']:
                    raw = str(val).strip()
                    found = re.findall(r'\b\d{6}\b', raw) 
                    if found: bat_keys.extend(found)
                    else: bat_keys.append(self.player_id_map.get(raw.lower(), raw))
                            
            for bowl_col_name in ['bowler_id', 'bowler', 'bowler_1_id']:
                val = row.get(bowl_col_name)
                if val is not None and str(val).strip().lower() not in ['nan', 'none', '', 'na', 'null']:
                    val_str = str(val).strip()
                    bowl_keys.append(self.player_id_map.get(val_str.lower(), val_str))
            
            if not bat_keys:
                for col_name, val in row.items():
                    col_lower = str(col_name).lower()
                    if col_lower not in ['id', 'matchid', 'match_id'] and ('bat' in col_lower or 'strike' in col_lower):
                        bat_keys.extend(re.findall(r'\b\d{6}\b', str(val)))
            
            if not bowl_keys:
                for col_name, val in row.items():
                    col_lower = str(col_name).lower()
                    if col_lower not in ['id', 'matchid', 'match_id'] and 'bowl' in col_lower:
                        bowl_keys.extend(re.findall(r'\b\d{6}\b', str(val)))
            
            bat_keys = list(set(bat_keys))
            bowl_keys = list(set(bowl_keys))
            
            num_batters = len(bat_keys) if len(bat_keys) > 0 else self.mean_batters 
            
            base_sr = 145.0 
            if inn == 2:
                low_target = self.modern_target * 0.85 
                high_target = self.modern_target * 1.10
                if 0 < target < low_target: base_sr = 135.0
                elif target >= high_target: base_sr = 155.0
            
            active_sr = base_sr
            active_econ = self.global_pp_econ 

            if bat_keys:
                known_srs = [self.player_sr_dict[pid] for pid in bat_keys if pid in self.player_sr_dict]
                if known_srs: active_sr = np.mean(known_srs)
                    
            if bowl_keys:
                known_econs = [self.player_econ_dict[pid] for pid in bowl_keys if pid in self.player_econ_dict]
                if known_econs: active_econ = np.mean(known_econs)
                
            active_venue = self.venue_form_dict.get((v, inn), self.mean_score if self.mean_score > 0 else 62.0)
            h2h_avg = self.h2h_dict.get((t_bat, t_bowl, inn), self.mean_score)
            
            base_data = {
                'bat_encoded': bat_enc, 'bowl_encoded': bowl_enc, 'venue_encoded': ven_enc, 'year': y,
                'is_modern': is_modern, 'top_order_sr': active_sr, 'top_bowler_econ': active_econ, 
                'recent_venue_avg': active_venue, 'num_batters': num_batters, 'h2h_avg': h2h_avg
            }
            
            try:
                if inn == 1:
                    X_test = pd.DataFrame([base_data])[self.features_inn1]
                    pred = self.model_inn1.predict(X_test).item()
                else:
                    base_data['target_score'] = target
                    X_test = pd.DataFrame([base_data])[self.features_inn2]
                    pred = self.model_inn2.predict(X_test).item()
                
                if num_batters >= 5:
                    extreme_pressure = 0.85 if num_batters == 5 else 0.72
                    pred = pred * extreme_pressure
                    
                pred = np.clip(pred, 25.0, 115.0) 
                
            except Exception: 
                if num_batters <= 2: w_p = 1.0      
                elif num_batters == 3: w_p = 0.92     
                elif num_batters == 4: w_p = 0.84 
                elif num_batters == 5: w_p = 0.85    
                else: w_p = 0.72 
                pred = np.clip((self.mean_score if self.mean_score > 0 else 62.0) * w_p, 25.0, 115.0)
                
            predictions.append({
                "id": row.get("id", row.get("matchId", row.get("match_id", row.name))), 
                "predicted_score": int(round(pred))
            })
            
        return pd.DataFrame(predictions)