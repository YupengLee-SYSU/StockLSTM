package org.deeplearning4j.examples.recurrent.indexlstm;

public class DailyData {

    private double Price;

	private double tips_5y;
    private double tips_10y;
    private double tips_20y;
    private double tips_30y;
    private double tips_long;
    private double ust_bill_10y;
    private double libor_overnight;
    private double spdr;
    private double usd_cny;

    public DailyData(){}

    public void setPrice(double Price){
        this.Price = Price;
    }
    public double getPrice(){
        return  Price;
    }

    public void settips_10y(double tips_10y){
        this.tips_10y = tips_10y;
    }
    public double gettips_10y(){
        return  tips_10y;
    }

    public void settips_20y(double tips_20y){
        this.tips_20y = tips_20y;
    }
    public double gettips_20y(){
        return  tips_20y;
    }

    public void settips_30y(double tips_30y){
        this.tips_30y = tips_30y;
    }
    public double gettips_30y(){
        return  tips_30y;
    }

    public void settips_long(double tips_long){
        this.tips_long = tips_long;
    }
    public double gettips_long(){
        return  tips_long;
    }

    public void setust_bill_10y(double ust_bill_10y){
        this.ust_bill_10y = ust_bill_10y;
    }
    public double getust_bill_10y(){
        return  ust_bill_10y;
    }

    public void setlibor_overnight(double libor_overnight){
        this.libor_overnight = libor_overnight;
    }
    public double getlibor_overnight(){
        return  libor_overnight;
    }

    public void setspdr(double spdr){
        this.spdr = spdr;
    }
    public double getspdr(){
        return  spdr;
    }

    public void setusd_cny(double usd_cny){
        this.usd_cny = usd_cny;
    }
    public double getusd_cny(){
        return  usd_cny;
    }

    public void setTIPS_5y(double TIPS_5y){
        this.tips_5y = TIPS_5y;
    }
    public double getTIPS_5y(){
        return  tips_5y;
    }
}
