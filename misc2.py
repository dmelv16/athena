           # Only flag anomalies for steady state
            if label == 'steady_state':
                flags = []
                reasons = []
                
                # Use dynamic thresholds if provided, otherwise use fixed
                if dynamic_thresholds:
                    # Use dynamic thresholds for this OFP/test_case group
                    group_key = f"{grouping.get('ofp', 'NA')}_{grouping.get('test_case', 'NA')}"
                    if group_key in dynamic_thresholds:
                        thresh = dynamic_thresholds[group_key]
                        
                        # Check all 4 thresholds
                        failed_checks = 0
                        
                        # Check variance
                        if ('min_variance' in thresh and metrics['variance'] < thresh['min_variance']) or \
                           ('max_variance' in thresh and metrics['variance'] > thresh['max_variance']):
                            failed_checks += 1
                            if 'min_variance' in thresh and metrics['variance'] < thresh['min_variance']:
                                reasons.append(f"Variance {metrics['variance']:.3f} < {thresh['min_variance']:.3f} (dynamic)")
                            else:
                                reasons.append(f"Variance {metrics['variance']:.3f} > {thresh['max_variance']:.3f} (dynamic)")
                        
                        # Check std
                        if ('min_std' in thresh and metrics['std'] < thresh['min_std']) or \
                           ('max_std' in thresh and metrics['std'] > thresh['max_std']):
                            failed_checks += 1
                            if 'min_std' in thresh and metrics['std'] < thresh['min_std']:
                                reasons.append(f"Std {metrics['std']:.3f} < {thresh['min_std']:.3f} (dynamic)")
                            else:
                                reasons.append(f"Std {metrics['std']:.3f} > {thresh['max_std']:.3f} (dynamic)")
                        
                        # Check abs_slope
                        if ('min_abs_slope' in thresh and metrics['abs_slope'] < thresh['min_abs_slope']) or \
                           ('max_abs_slope' in thresh and metrics['abs_slope'] > thresh['max_abs_slope']):
                            failed_checks += 1
                            if 'min_abs_slope' in thresh and metrics['abs_slope'] < thresh['min_abs_slope']:
                                reasons.append(f"Slope {metrics['abs_slope']:.4f} < {thresh['min_abs_slope']:.4f} (dynamic)")
                            else:
                                reasons.append(f"Slope {metrics['abs_slope']:.4f} > {thresh['max_abs_slope']:.4f} (dynamic)")
                        
                        # Check IQR
                        if ('min_iqr' in thresh and metrics['iqr'] < thresh['min_iqr']) or \
                           ('max_iqr' in thresh and metrics['iqr'] > thresh['max_iqr']):
                            failed_checks += 1
                            if 'min_iqr' in thresh and metrics['iqr'] < thresh['min_iqr']:
                                reasons.append(f"IQR {metrics['iqr']:.3f} < {thresh['min_iqr']:.3f} (dynamic)")
                            else:
                                reasons.append(f"IQR {metrics['iqr']:.3f} > {thresh['max_iqr']:.3f} (dynamic)")
                        
                        # Only flag if ALL 4 fail
                        if failed_checks == 4:
                            flags.append('all_thresholds_failed')
                            # Keep all the reasons to show what failed
                        else:
                            # Clear reasons if not flagging
                            reasons = []
                            
                else:
                    # Use fixed thresholds (original logic) - but require all to fail
                    failed_checks = 0
                    
                    if metrics['variance'] > self.steady_state_thresholds['max_variance']:
                        failed_checks += 1
                        reasons.append(f"Variance {metrics['variance']:.3f} > {self.steady_state_thresholds['max_variance']}")
                    
                    if metrics['std'] > self.steady_state_thresholds['max_std']:
                        failed_checks += 1
                        reasons.append(f"Std {metrics['std']:.3f} > {self.steady_state_thresholds['max_std']}")
                    
                    if metrics['abs_slope'] > self.steady_state_thresholds['max_slope']:
                        failed_checks += 1
                        reasons.append(f"Slope {metrics['abs_slope']:.4f} > {self.steady_state_thresholds['max_slope']}")
                    
                    # Only flag if all fail (3 for fixed since no IQR threshold)
                    if failed_checks >= 3:
                        flags.append('all_thresholds_failed')
                    else:
                        reasons = []
                
                # NO Z-SCORE CHECKING
                metrics['n_outliers_zscore'] = 0
                metrics['max_zscore'] = 0
                metrics['outlier_indices'] = []
                metrics['n_outliers_iqr'] = 0
                
                metrics['flagged'] = len(flags) > 0
                metrics['flags'] = ', '.join(flags) if flags else ''
                metrics['flag_reasons'] = '; '.join(reasons) if reasons else ''
                
            else:
                # Other labels (de-energized, stabilizing, unidentified) are NEVER flagged
                metrics['flagged'] = False
                metrics['flags'] = ''
                metrics['flag_reasons'] = ''
                metrics['n_outliers_zscore'] = 0
                metrics['n_outliers_iqr'] = 0
                metrics['max_zscore'] = 0
                metrics['outlier_indices'] = []
            
            results.append(metrics)
        
        return results, df, grouping
        
    except Exception as e:
        self.failed_files.append((csv_path.name, str(e)))
        return None
