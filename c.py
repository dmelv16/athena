# Prepare data for JavaScript
        flips_data = []
        # -----------------------------------------------------------------
        # FIX: Combine both flips with changes and flips with no changes
        # -----------------------------------------------------------------
        if self.df_flips is not None or self.df_flips_no_changes is not None:
            df_all_flips = pd.concat([
                self.df_flips if self.df_flips is not None else pd.DataFrame(),
                self.df_flips_no_changes if self.df_flips_no_changes is not None else pd.DataFrame()
            ], ignore_index=True)

            if not df_all_flips.empty:
                df_temp = df_all_flips.copy()
                for col in ['timestamp_busA', 'timestamp_busB']:
                    if col in df_temp.columns:
                        df_temp[col] = df_temp[col].astype(str)
                flips_data = df_temp.to_dict('records')
