package com.saunak.milan;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
  static final String PREFS="milan_mobile", KEY="server_url";
  EditText url; TextView status; Button open;
  int dp(int v){return (int)(v*getResources().getDisplayMetrics().density+.5f);}
  @Override public void onCreate(Bundle b){super.onCreate(b); getWindow().setStatusBarColor(Color.rgb(5,10,17)); setContentView(ui()); String s=getSharedPreferences(PREFS,0).getString(KEY,""); url.setText(s);}
  LinearLayout.LayoutParams lp(){return new LinearLayout.LayoutParams(-1,-2);}
  android.view.View ui(){LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.VERTICAL);r.setGravity(Gravity.CENTER_HORIZONTAL);r.setPadding(dp(24),dp(60),dp(24),dp(24));r.setBackgroundColor(Color.rgb(5,10,17)); TextView h=new TextView(this);h.setText("◉  MILAN");h.setTextColor(Color.rgb(113,246,226));h.setTextSize(30);h.setGravity(Gravity.CENTER);r.addView(h,lp());TextView sub=new TextView(this);sub.setText("PERSONAL COGNITIVE OS\nANDROID GATEWAY");sub.setTextColor(Color.WHITE);sub.setTextSize(15);sub.setGravity(Gravity.CENTER);sub.setPadding(0,dp(14),0,dp(34));r.addView(sub,lp());url=new EditText(this);url.setHint("https://milan.your-domain.com");url.setSingleLine(true);url.setTextColor(Color.WHITE);url.setHintTextColor(Color.GRAY);url.setBackgroundColor(Color.rgb(11,22,34));url.setPadding(dp(16),dp(14),dp(16),dp(14));r.addView(url,lp());open=new Button(this);open.setText("OPEN MILAN");open.setOnClickListener(v->check());r.addView(open,lp());Button direct=new Button(this);direct.setText("OPEN WITHOUT HEALTH CHECK");direct.setOnClickListener(v->{String s=norm(url.getText().toString());if(s!=null){save(s);launch(s);}else msg("Enter a valid Milan URL",true);});r.addView(direct,lp());status=new TextView(this);status.setText("Enter your stable Milan HTTPS URL once. Provider secrets stay on your server, never in this APK.");status.setTextColor(Color.LTGRAY);status.setGravity(Gravity.CENTER);status.setPadding(0,dp(22),0,0);r.addView(status,lp());return r;}
  String norm(String s){if(s==null)return null;s=s.trim();while(s.endsWith("/"))s=s.substring(0,s.length()-1);if(!s.startsWith("https://"))return null;try{Uri u=Uri.parse(s);return u.getHost()==null?null:s;}catch(Exception e){return null;}}
  void check(){String s=norm(url.getText().toString());if(s==null){msg("Use a valid HTTPS Milan URL",true);return;}open.setEnabled(false);msg("Checking Milan…",false);Executors.newSingleThreadExecutor().execute(()->{boolean ok=false;String d="";try{HttpURLConnection c=(HttpURLConnection)new URL(s+"/health").openConnection();c.setConnectTimeout(5000);c.setReadTimeout(5000);int code=c.getResponseCode();ok=code>=200&&code<300;d="HTTP "+code;c.disconnect();}catch(Exception e){d=e.getClass().getSimpleName();}boolean f=ok;String fd=d;runOnUiThread(()->{open.setEnabled(true);if(f){save(s);msg("Milan online. Opening…",false);launch(s);}else msg("Health check failed ("+fd+"). Is the laptop/tunnel online?",true);});});}
  void save(String s){getSharedPreferences(PREFS,0).edit().putString(KEY,s).apply();}
  void launch(String s){startActivity(new Intent(Intent.ACTION_VIEW,Uri.parse(s)));}
  void msg(String s,boolean err){status.setText(s);status.setTextColor(err?Color.rgb(255,150,120):Color.LTGRAY);}
}
