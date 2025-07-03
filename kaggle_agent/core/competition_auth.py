"""Competition authentication handler."""

import webbrowser
import time
from typing import Optional


class CompetitionAuth:
    """Handle competition authentication and rules acceptance."""
    
    def __init__(self):
        self.accepted_competitions = set()
    
    def ensure_competition_access(self, competition_name: str, auto_open: bool = True) -> bool:
        """Ensure access to competition by handling authentication."""
        if competition_name in self.accepted_competitions:
            return True
        
        print(f"\n📋 Competition Authentication Check: {competition_name}")
        
        # First, try to access the competition
        try:
            import kaggle
            # Try to list files - this will fail if not authenticated
            kaggle.api.competition_list_files(competition_name)
            print("✓ Competition access verified")
            self.accepted_competitions.add(competition_name)
            return True
            
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                return self._handle_forbidden_access(competition_name, auto_open)
            else:
                print(f"✗ Unexpected error: {e}")
                return False
    
    def _handle_forbidden_access(self, competition_name: str, auto_open: bool) -> bool:
        """Handle 403 Forbidden errors."""
        print("\n⚠️  Competition requires authentication or rules acceptance")
        print(f"\nThis competition requires you to:")
        print("1. Be logged in to Kaggle")
        print("2. Accept the competition rules")
        print("3. Join the competition (if required)")
        
        if auto_open:
            self._open_competition_page(competition_name)
        else:
            print(f"\nPlease visit: https://www.kaggle.com/c/{competition_name}/rules")
        
        print("\n" + "="*50)
        print("ACTION REQUIRED:")
        print("1. Accept the competition rules in your browser")
        print("2. Return here and press Enter to continue")
        print("="*50)
        
        input("\nPress Enter after accepting rules...")
        
        # Verify access again
        try:
            import kaggle
            kaggle.api.competition_list_files(competition_name)
            print("\n✓ Competition access granted!")
            self.accepted_competitions.add(competition_name)
            return True
        except:
            print("\n✗ Still unable to access competition")
            print("Please ensure you have:")
            print("- Accepted all competition rules")
            print("- Joined the competition if required")
            print("- Verified your phone number if required")
            
            retry = input("\nRetry? (y/n): ")
            if retry.lower() == 'y':
                return self.ensure_competition_access(competition_name, auto_open=False)
            return False
    
    def _open_competition_page(self, competition_name: str):
        """Open competition page in browser."""
        url = f"https://www.kaggle.com/c/{competition_name}/rules"
        print(f"\n🌐 Opening competition page in browser...")
        print(f"   URL: {url}")
        
        try:
            webbrowser.open(url)
            time.sleep(2)  # Give browser time to open
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
    
    def check_competition_requirements(self, competition_name: str) -> dict:
        """Check competition requirements."""
        requirements = {
            'rules_accepted': False,
            'phone_verified': None,
            'team_merge_deadline': None,
            'gpu_required': False,
            'internet_access': True
        }
        
        try:
            import kaggle
            # Get competition details
            competitions = kaggle.api.competitions_list(search=competition_name)
            for comp in competitions:
                if comp.ref == competition_name or competition_name in comp.ref:
                    # Check various requirements
                    if hasattr(comp, 'phoneVerificationRequired'):
                        requirements['phone_verified'] = comp.phoneVerificationRequired
                    if hasattr(comp, 'mergingDeadline'):
                        requirements['team_merge_deadline'] = comp.mergingDeadline
                    
                    # Try to access files to check if rules accepted
                    try:
                        kaggle.api.competition_list_files(competition_name)
                        requirements['rules_accepted'] = True
                    except:
                        pass
                    
                    break
        except:
            pass
        
        return requirements