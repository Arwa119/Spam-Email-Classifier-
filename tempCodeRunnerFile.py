if hasattr(model, 'coef_'):
#     # Get feature importance
#     importance = model.coef_[0]
    
#     # Create DataFrame with feature names and importance
#     feature_importance = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': importance
#     })
    
#     # Sort by absolute importance
#     feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
#     feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
    
#     print("\nTop 20 most important words for classification:")
#     print(feature_importance.head(20))
    
#     # Plot feature importance
#     plt.figure(figsize=(12, 8))
#     top_features = feature_importance.head(20)
#     sns.barplot(x='Importance', y='Feature', data=top_features)
#     plt.title('Top 20 Most Important Words for Email Classification')
#     plt.tight_layout()
#     plt.savefig('feature_importance.png')
#     plt.close()